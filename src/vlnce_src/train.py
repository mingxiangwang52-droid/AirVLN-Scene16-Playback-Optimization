import os
import sys
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy
import airsim

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video
from utils.action_guard import RepeatActionGuard
from utils.instruction_segments import SegmentInstructionState
from utils.segment_training import SegmentTrainingAdapter
from utils.segment_grounding_controller import SegmentGroundingController
from utils.depth_safety import DepthSafetyGuard
from airsim_plugin.airsim_settings import AirsimActions

from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer


def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False


def _trajectory_action_weights(oracle_actions, inflec_weights):
    """Weight rare segment exits without letting long vertical runs dominate."""
    inflections = torch.cat(
        [
            torch.tensor([1], dtype=torch.long),
            (oracle_actions[1:] != oracle_actions[:-1]).long(),
        ]
    )
    weights = inflec_weights[inflections]

    stop_weight = max(float(getattr(args, "stop_weight_coef", 1.0)), 0.0)
    if stop_weight != 1.0:
        weights = weights * torch.where(
            oracle_actions == 0,
            torch.as_tensor(stop_weight, dtype=weights.dtype),
            torch.ones((), dtype=weights.dtype),
        )

    repeat_weight = max(float(getattr(args, "vertical_repeat_weight", 1.0)), 0.0)
    grace_steps = max(int(getattr(args, "vertical_repeat_grace_steps", 0)), 0)
    if repeat_weight != 1.0 and oracle_actions.numel() > 1:
        repeated = torch.zeros_like(oracle_actions, dtype=torch.bool)
        run_length = 1
        for index in range(1, oracle_actions.numel()):
            current = int(oracle_actions[index].item())
            previous = int(oracle_actions[index - 1].item())
            if current == previous and current in (4, 5):
                run_length += 1
                if run_length > grace_steps:
                    repeated[index] = True
            else:
                run_length = 1
        weights = weights * torch.where(
            repeated,
            torch.as_tensor(repeat_weight, dtype=weights.dtype),
            torch.ones((), dtype=weights.dtype),
        )

    return weights


def _set_eval_instruction_segment(train_env, batch_index, segment_text, tokenizer):
    tokens = tokenizer.encode_sentence(segment_text)
    if tokens is None:
        raise ValueError(f"failed to tokenize instruction segment: {segment_text!r}")
    for episodes in (train_env.batch, train_env.VectorEnvUtil.batch):
        episodes[batch_index]["instruction"]["instruction_tokens"] = np.copy(tokens)
        episodes[batch_index]["instruction"]["active_segment_text"] = segment_text
    return tokens


class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning("END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i+1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        return (
            obs,
            prev_actions,
            oracle_actions,
            _trajectory_action_weights(oracle_actions, self.inflec_weights),
        )

    def __iter__(self):
        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )

        return self


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
        holdout_fraction=0.0,
        holdout_seed=20260719,
        holdout_partition="all",
        segment_adapter=None,
        source_limit=0,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self._segment_preload = []
        self.batch_size = batch_size
        self.segment_adapter = segment_adapter

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        source_limit = max(0, int(source_limit))
        if source_limit > 0 and len(self.keys) > source_limit:
            indices = np.linspace(
                0, len(self.keys) - 1, source_limit, dtype=np.int64
            )
            self.keys = [self.keys[index] for index in indices]

        holdout_fraction = max(0.0, min(float(holdout_fraction), 0.5))
        if holdout_fraction > 0.0:
            rng = np.random.RandomState(int(holdout_seed))
            ordering = rng.permutation(len(self.keys)).tolist()
            holdout_count = max(1, int(round(len(self.keys) * holdout_fraction)))
            holdout_indices = set(ordering[:holdout_count])
            if holdout_partition == "train":
                self.keys = [key for index, key in enumerate(self.keys) if index not in holdout_indices]
            elif holdout_partition == "holdout":
                self.keys = [key for index, key in enumerate(self.keys) if index in holdout_indices]
            else:
                raise ValueError(f"unknown holdout partition: {holdout_partition}")

        self.source_length = len(self.keys)
        self.length = (
            self.segment_adapter.expanded_length(self.keys)
            if self.segment_adapter is not None
            else self.source_length
        )

        self.iter_start = 0
        self.iter_end = self.source_length
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i+1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i+1, self.preload_size))

                    key = self.keys[self.load_ordering.pop()]
                    new_preload.append((
                        key,
                        msgpack_numpy.unpackb(
                            txn.get(str(key).encode()),
                            raw=False,
                        ),
                    ))

                    lengths.append(len(new_preload[-1][1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        if self._segment_preload:
            obs, prev_actions, oracle_actions = self._segment_preload.pop()
        else:
            episode_id, packed = self._load_next()
            obs, prev_actions, oracle_actions = packed
            if self.segment_adapter is not None:
                self._segment_preload = list(
                    reversed(
                        self.segment_adapter.split_trajectory(
                            episode_id, obs, prev_actions, oracle_actions
                        )
                    )
                )
                obs, prev_actions, oracle_actions = self._segment_preload.pop()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        return (
            obs,
            prev_actions,
            oracle_actions,
            _trajectory_action_weights(oracle_actions, self.inflec_weights),
        )

    def __iter__(self):
        self._segment_preload = []
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None:
            start = 0
            end = self.source_length
        else:
            per_worker = int(np.ceil(self.source_length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.source_length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


class InterleavedTrajectoryDataset(torch.utils.data.IterableDataset):
    """Interleave a large primary LMDB with a smaller replay LMDB."""

    def __init__(self, primary, replay, primary_per_replay=0):
        super().__init__()
        self.primary = primary
        self.replay = replay
        self.primary_length = primary.length
        self.replay_length = replay.length
        self.batch_size = primary.batch_size
        self.repeat_replay = int(primary_per_replay) > 0
        if self.repeat_replay:
            self.primary_per_replay = max(1, int(primary_per_replay))
            replay_samples = int(math.ceil(self.primary_length / self.primary_per_replay))
            self.length = self.primary_length + replay_samples
        else:
            self.primary_per_replay = max(
                1,
                int(round(self.primary_length / max(1, self.replay_length))),
            )
            self.length = self.primary_length + self.replay_length

    def __iter__(self):
        primary_iter = iter(self.primary)
        replay_iter = iter(self.replay)
        primary_done = False
        replay_done = False

        while not primary_done:
            yielded_primary = 0
            for _ in range(self.primary_per_replay):
                if primary_done:
                    break
                try:
                    yield next(primary_iter)
                    yielded_primary += 1
                except StopIteration:
                    primary_done = True

            if yielded_primary == 0:
                break

            if not replay_done:
                try:
                    yield next(replay_iter)
                except StopIteration:
                    if self.repeat_replay and not primary_done:
                        replay_iter = iter(self.replay)
                        try:
                            yield next(replay_iter)
                        except StopIteration:
                            replay_done = True
                    else:
                        replay_done = True


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)
    trajectory_lengths = [min(item.size(0), 500) for item in prev_actions_batch]

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(trajectory_lengths)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    trajectory_valid = torch.zeros(max_traj_len, B, 1, dtype=torch.float32)
    for bid, length in enumerate(trajectory_lengths):
        trajectory_valid[:length, bid] = 1.0
    observations_batch["trajectory_valid"] = trajectory_valid.view(-1, 1)

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    return tok


def initialize_env(split='train'):
    tok = initialize_tokenizer()

    train_env = AirVLNENV(batch_size=args.batchSize, split=split, tokenizer=tok)

    return train_env


def initialize_trainer():
    from gym import spaces
    from airsim_plugin.airsim_settings import AirsimActions

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(int(len(AirsimActions)))

    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "").strip()
    trainer = VLNCETrainer(
        load_from_ckpt=bool(init_checkpoint),
        observation_space=observation_space,
        action_space=action_space,
        ckpt_path=init_checkpoint or None,
    )
    if init_checkpoint:
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = float(args.lr)
        logger.info("Fine-tuning from checkpoint %s with lr=%s", init_checkpoint, args.lr)

    logger.info('initialize_trainer over')
    return trainer


def collect_data(data_it=0):
    logger.info(args)

    train_env = initialize_env(split=args.TRAIN_SPLIT)
    trainer = initialize_trainer()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

    def hook_builder(tgt_tensor):
        def hook(m, i, o):
            tgt_tensor.set_(o.cpu())

        return hook

    rgb_features = torch.zeros((1,), device="cpu")
    if not args.ablate_rgb:
        rgb_hook = trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            hook_builder(rgb_features)
        )
    else:
        rgb_hook = None

    depth_features = torch.zeros((1,), device="cpu")
    if not args.ablate_depth:
        depth_hook = trainer.policy.net.depth_encoder.visual_encoder.register_forward_hook(
            hook_builder(depth_features)
        )
    else:
        depth_hook = None

    beta = min(max(float(args.collect_beta), 0.0), 1.0)
    logger.warning("Collection teacher-action beta=%s", beta)


    #
    with torch.no_grad():
        end_iter = len(train_env.data)
        pbar = None
        pbar_pre_index = 0
        while train_env.index_data < end_iter:
            pbar_pre_index = train_env.index_data
            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if pbar is None:
                pbar = tqdm.tqdm(total=end_iter)
                pbar.update(train_env.index_data)
            else:
                pbar.update(n=train_env.index_data-pbar_pre_index)

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)

            ended = False

            for t in range(int(args.maxAction) + 1):
                logger.info('{} - {} / {}'.format(int(train_env.index_data)-int(train_env.batch_size), t, end_iter))

                for i in range(train_env.batch_size):
                    if dones[i] and not skips[i]:
                        if len(episodes[i]) == 0:
                            logger.warning(
                                'Skipping zero-step episode %s after reset',
                                infos[i].get('episode_id', 'unknown'),
                            )
                            envs_to_pause.append(i)
                            skips[i] = True
                            continue
                        if args.collect_type in ['TF']:
                            _episodes = episodes[i].copy()
                            for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                                for __i, __j in enumerate(_episodes):
                                    _episodes[__i][0]['instruction'] = _j

                                ep = _episodes.copy()
                                traj_obs = batch_obs(
                                    [step[0] for step in ep],
                                    device=torch.device("cpu"),
                                )
                                traj_obs.pop('teacher_action', None)
                                for k, v in traj_obs.items():
                                    traj_obs[k] = v.numpy()

                                transposed_ep = [
                                    traj_obs,
                                    np.array([step[1] for step in ep], dtype=np.int64),
                                    np.array([step[2] for step in ep], dtype=np.int64),
                                ]

                                train_env.threading_lock_lmdb_features_txn.acquire()
                                lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                                train_env.lmdb_features_txn.put(
                                    lmdb_key.encode(),
                                    msgpack_numpy.packb(
                                        transposed_ep, use_bin_type=True
                                    ),
                                )
                                train_env.lmdb_features_txn.commit()
                                train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                                train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                                train_env.threading_lock_lmdb_features_txn.release()
                                logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            _episodes = []
                            envs_to_pause.append(i)
                            skips[i] = True

                        else:
                            ep = episodes[i]
                            traj_obs = batch_obs(
                                [step[0] for step in ep],
                                device=torch.device("cpu"),
                            )
                            traj_obs.pop('teacher_action', None)
                            for k, v in traj_obs.items():
                                traj_obs[k] = v.numpy()

                            transposed_ep = [
                                traj_obs,
                                np.array([step[1] for step in ep], dtype=np.int64),
                                np.array([step[2] for step in ep], dtype=np.int64),
                            ]

                            train_env.threading_lock_lmdb_features_txn.acquire()
                            lmdb_key = str(infos[i]['episode_id'])
                            train_env.lmdb_features_txn.put(
                                lmdb_key.encode(),
                                msgpack_numpy.packb(
                                    transposed_ep, use_bin_type=True
                                ),
                            )
                            train_env.lmdb_features_txn.commit()
                            train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                            train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                            train_env.lmdb_collected_keys.add(lmdb_key)
                            train_env.threading_lock_lmdb_features_txn.release()
                            logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            envs_to_pause.append(i)
                            skips[i] = True

                    if np.array(dones).all():
                        ended = True

                if ended:
                    break

                try:
                    actions, rnn_states = trainer.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )
                except RuntimeError as e:
                    logger.warning(f'Skipping batch at index {train_env.index_data} due to: {e}')
                    skips = [True] * train_env.batch_size
                    break
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch['teacher_action'].long(),
                    actions,
                )
                if args.collect_prevent_early_stop:
                    premature_stop = (actions == AirsimActions.STOP) & (
                        batch['teacher_action'].long() != AirsimActions.STOP
                    )
                    actions = torch.where(
                        premature_stop,
                        batch['teacher_action'].long(),
                        actions,
                    )

                for i in range(train_env.batch_size):
                    if not args.ablate_rgb and rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if not args.ablate_depth and depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if i in envs_to_pause:
                        continue

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch['teacher_action'][i].item(),
                        )
                    )

                prev_actions.copy_(actions)

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

            for i in range(train_env.batch_size):
                if dones[i] and not t >= int(args.maxAction):
                    continue

                if args.collect_type in ['TF']:
                    _episodes = episodes[i].copy()
                    for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                        for __i, __j in enumerate(_episodes):
                            _episodes[__i][0]['instruction'] = _j

                        ep = _episodes.copy()
                        if len(ep) <= 0:
                            continue
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        traj_obs.pop('teacher_action', None)
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]

                        train_env.threading_lock_lmdb_features_txn.acquire()
                        lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                        train_env.lmdb_features_txn.put(
                            lmdb_key.encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )
                        train_env.lmdb_features_txn.commit()
                        train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                        train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                        train_env.lmdb_collected_keys.add(lmdb_key)
                        train_env.threading_lock_lmdb_features_txn.release()
                        logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    _episodes = []
                    envs_to_pause.append(i)
                    skips[i] = True

                else:
                    ep = episodes[i]
                    if len(ep) <= 0:
                        continue
                    traj_obs = batch_obs(
                        [step[0] for step in ep],
                        device=torch.device("cpu"),
                    )
                    traj_obs.pop('teacher_action', None)
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()

                    transposed_ep = [
                        traj_obs,
                        np.array([step[1] for step in ep], dtype=np.int64),
                        np.array([step[2] for step in ep], dtype=np.int64),
                    ]

                    train_env.threading_lock_lmdb_features_txn.acquire()
                    lmdb_key = str(infos[i]['episode_id'])
                    train_env.lmdb_features_txn.put(
                        lmdb_key.encode(),
                        msgpack_numpy.packb(
                            transposed_ep, use_bin_type=True
                        ),
                    )
                    train_env.lmdb_features_txn.commit()
                    train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                    train_env.lmdb_collected_keys.add(lmdb_key)
                    train_env.threading_lock_lmdb_features_txn.release()
                    logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    envs_to_pause.append(i)
                    skips[i] = True

    try:
        pbar.close()
    except:
        pass

    if rgb_hook is not None:
        rgb_hook.remove()
    if depth_hook is not None:
        depth_hook.remove()

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass
    logger.info('END data_it: {}'.format(data_it))


def train_vlnce():
    logger.info(args)

    if get_rank() == 0:
        writer = SummaryWriter(
            log_dir=str(Path(args.project_prefix) / 'DATA/output/{}/train/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        )
    else:
        writer = None

    trainer = initialize_trainer()

    for dagger_it in range(int(args.dagger_it)):
        step_id = 0

        if torch.cuda.is_available():
            with torch.cuda.device(trainer.device):
                torch.cuda.empty_cache()
        gc.collect()

        lmdb_features_dir = str(Path(args.project_prefix) / 'DATA/img_features/collect/{}/{}'.format(args.name, args.TRAIN_SPLIT))
        assert os.path.exists(str(lmdb_features_dir))
        holdout_dataset = None
        holdout_diter = None
        if args.DistributedDataParallel:
            dataset = DDPIWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
            )
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )
        else:
            holdout_fraction = float(getattr(args, "overfit_holdout_fraction", 0.0))
            replay_feature_dir = str(
                getattr(args, "replay_feature_dir", "") or ""
            ).strip()
            segment_adapter = None
            if bool(getattr(args, "segment_train_instructions", False)):
                segment_data_path = str(
                    getattr(args, "segment_train_data_path", "") or ""
                ).strip()
                if not segment_data_path:
                    segment_data_path = str(
                        Path(args.TRAIN_VOCAB).parent / f"{args.TRAIN_SPLIT}.json"
                    )
                segment_adapter = SegmentTrainingAdapter(
                    segment_data_path,
                    initialize_tokenizer(),
                    max_segments_per_trajectory=int(
                        args.segment_train_max_segments_per_trajectory
                    ),
                )
                logger.warning(
                    "Segment training enabled with %d episode mappings from %s",
                    len(segment_adapter.segment_tokens),
                    segment_data_path,
                )

            def make_dataset(partition):
                primary = IWTrajectoryDataset(
                    lmdb_features_dir,
                    use_iw=True,
                    inflection_weight_coef=float(args.inflection_weight_coef),
                    lmdb_map_size=5.0e12,
                    batch_size=args.batchSize,
                    holdout_fraction=holdout_fraction,
                    holdout_seed=int(args.overfit_holdout_seed),
                    holdout_partition=partition,
                    segment_adapter=segment_adapter,
                    source_limit=int(args.train_source_limit),
                )
                if not replay_feature_dir:
                    return primary
                if not os.path.isdir(replay_feature_dir):
                    raise FileNotFoundError(
                        f"replay feature directory not found: {replay_feature_dir}"
                    )
                replay = IWTrajectoryDataset(
                    replay_feature_dir,
                    use_iw=True,
                    inflection_weight_coef=float(args.inflection_weight_coef),
                    lmdb_map_size=5.0e12,
                    batch_size=args.batchSize,
                    holdout_fraction=holdout_fraction,
                    holdout_seed=int(args.overfit_holdout_seed),
                    holdout_partition=partition,
                    source_limit=0,
                )
                mixed = InterleavedTrajectoryDataset(
                    primary,
                    replay,
                    primary_per_replay=int(args.replay_primary_per_replay),
                )
                logger.warning(
                    "Interleaved dataset primary=%d replay=%d stride=%d",
                    mixed.primary_length,
                    mixed.replay_length,
                    mixed.primary_per_replay,
                )
                return mixed

            train_partition = "train" if holdout_fraction > 0 else "all"
            dataset = make_dataset(train_partition)
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )
            if holdout_fraction > 0:
                holdout_dataset = make_dataset("holdout")
                holdout_diter = torch.utils.data.DataLoader(
                    holdout_dataset,
                    batch_size=args.batchSize,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=False,
                    num_workers=0,
                )

        metrics_dir = Path(args.project_prefix) / 'DATA/output' / args.name / 'train' / 'metrics' / args.make_dir_time
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / 'overfit_metrics.jsonl'

        AuxLosses.activate()
        for epoch in tqdm.trange(int(args.epochs), dynamic_ncols=True):
            args.segment_align_current_epoch = epoch
            batch_cnt = 0
            train_loss_sum = 0.0
            train_action_loss_sum = 0.0
            train_aux_loss_sum = 0.0
            for batch in tqdm.tqdm(
                diter,
                total=dataset.length // dataset.batch_size if not args.DistributedDataParallel else (dataset.iter_end - dataset.iter_start) // dataset.batch_size,
                leave=False,
                dynamic_ncols=True,
            ):
                (
                    observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    weights_batch,
                ) = batch

                observations_batch = {
                    k: v.to(
                        device=trainer.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    for k, v in observations_batch.items()
                }

                loss, action_loss, aux_loss = trainer._update_agent(
                    observations_batch,
                    prev_actions_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                    not_done_masks.to(
                        device=trainer.device, non_blocking=True
                    ),
                    corrected_actions_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                    weights_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                )
                train_loss_sum += float(loss)
                train_action_loss_sum += float(action_loss)
                train_aux_loss_sum += float(aux_loss)

                logger.warning(
                    'dagger_it: {} / {} \t epoch: {} / {} \t batch: {} / {}'.format(
                        dagger_it, args.dagger_it,
                        epoch, args.epochs,
                        batch_cnt, dataset.length // dataset.batch_size
                    )
                )

                logger.info(f"train_loss: {loss}")
                logger.info(f"train_action_loss: {action_loss}")
                logger.info(f"train_aux_loss: {aux_loss}")
                logger.info(f"Batches processed: {step_id}.")
                logger.info(
                    f"On DAgger iter {dagger_it}, Epoch {epoch}."
                )
                logger.info('\n')

                if get_rank() == 0:
                    writer.add_scalar(
                        f"train_loss_iter_{dagger_it}", loss, step_id
                    )
                    writer.add_scalar(
                        f"train_action_loss_iter_{dagger_it}",
                        action_loss,
                        step_id,
                    )
                    writer.add_scalar(
                        f"train_aux_loss_iter_{dagger_it}",
                        aux_loss,
                        step_id,
                    )

                step_id += 1
                batch_cnt += 1

            holdout_loss = None
            holdout_action_loss = None
            holdout_aux_loss = None
            holdout_batches = 0
            if holdout_diter is not None:
                holdout_loss_sum = 0.0
                holdout_action_loss_sum = 0.0
                holdout_aux_loss_sum = 0.0
                trainer.policy.eval()
                with torch.no_grad():
                    for holdout_batch in tqdm.tqdm(
                        holdout_diter,
                        total=max(1, holdout_dataset.length // holdout_dataset.batch_size),
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        (
                            holdout_observations,
                            holdout_prev_actions,
                            holdout_not_done_masks,
                            holdout_corrected_actions,
                            holdout_weights,
                        ) = holdout_batch
                        holdout_observations = {
                            key: value.to(
                                device=trainer.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            for key, value in holdout_observations.items()
                        }
                        val_loss, val_action_loss, val_aux_loss = trainer._update_agent(
                            holdout_observations,
                            holdout_prev_actions.to(device=trainer.device, non_blocking=True),
                            holdout_not_done_masks.to(device=trainer.device, non_blocking=True),
                            holdout_corrected_actions.to(device=trainer.device, non_blocking=True),
                            holdout_weights.to(device=trainer.device, non_blocking=True),
                            backward=False,
                        )
                        holdout_loss_sum += float(val_loss)
                        holdout_action_loss_sum += float(val_action_loss)
                        holdout_aux_loss_sum += float(val_aux_loss)
                        holdout_batches += 1
                trainer.policy.train()
                holdout_loss = holdout_loss_sum / max(1, holdout_batches)
                holdout_action_loss = holdout_action_loss_sum / max(1, holdout_batches)
                holdout_aux_loss = holdout_aux_loss_sum / max(1, holdout_batches)

            epoch_metrics = {
                "epoch": int(epoch),
                "train_batches": int(batch_cnt),
                "holdout_batches": int(holdout_batches),
                "train_loss": train_loss_sum / max(1, batch_cnt),
                "train_action_loss": train_action_loss_sum / max(1, batch_cnt),
                "train_aux_loss": train_aux_loss_sum / max(1, batch_cnt),
                "holdout_loss": holdout_loss,
                "holdout_action_loss": holdout_action_loss,
                "holdout_aux_loss": holdout_aux_loss,
                "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                "train_samples": int(dataset.length),
                "holdout_samples": int(holdout_dataset.length) if holdout_dataset is not None else 0,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            logger.warning("OVERFIT_METRICS %s", json.dumps(epoch_metrics, ensure_ascii=False))
            if writer is not None and holdout_loss is not None:
                writer.add_scalar("holdout_loss", holdout_loss, epoch)
                writer.add_scalar("holdout_action_loss", holdout_action_loss, epoch)

            if is_main_process():
                if holdout_diter is not None or ((dagger_it * args.epochs + epoch)+1) % 5 == 0:
                    trainer.save_checkpoint(
                        f"ckpt.{dagger_it * args.epochs + epoch}.pth",
                        dagger_it,
                        epoch,
                    )

            if is_dist_avail_and_initialized() == 1:
                dist.barrier()

        if is_main_process():
            trainer.save_checkpoint(
                f"ckpt.LAST.pth",
                dagger_it,
                epoch,
            )
        AuxLosses.deactivate()
        if hasattr(args, "segment_align_current_epoch"):
            delattr(args, "segment_align_current_epoch")


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    tok = initialize_tokenizer()

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), 'The eval file/folder does not exist'
    if os.path.isfile(args.EVAL_CKPT_PATH_DIR):
        from Model.utils.common import get_checkpoint_id

        # evaluate singe checkpoint
        proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
        if proposed_index is not None:
            ckpt_idx = proposed_index
        else:
            ckpt_idx = 100000

        _eval_checkpoint(
            checkpoint_path=args.EVAL_CKPT_PATH_DIR,
            writer=writer,
            tok=tok,
            checkpoint_index=ckpt_idx,
        )
        logger.info("END evaluate")
    else:
        from Model.utils.common import poll_checkpoint_folder

        # evaluate multiple checkpoints in order
        prev_ckpt_ind = -1
        retry_cnt = 0
        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                )
                if current_ckpt is None:
                    retry_cnt += 1
                    if retry_cnt > 5:
                        break
                    time.sleep(2)
            if current_ckpt is None:
                break
            logger.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += 1
            retry_cnt = 0

            _eval_checkpoint(
                checkpoint_path=current_ckpt,
                writer=writer,
                tok=tok,
                checkpoint_index=prev_ckpt_ind,
            )

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(
    checkpoint_path: str,
    writer,
    tok,
    checkpoint_index: int = 0,
) -> None:
    logger.info(f"checkpoint_path: {checkpoint_path}")


    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        train_env = AirVLNENV(batch_size=args.batchSize, split=args.EVAL_DATASET, tokenizer=tok)


    #
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return


    #
    trainer = VLNCETrainer(
        load_from_ckpt=True,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        ckpt_path=checkpoint_path,
    )
    trainer.policy.eval()

    visual_grounding_features = {"rgb": None, "depth": None}
    visual_grounding_hooks = []
    if args.EVAL_learned_segment_grounding:
        def cache_visual_feature(name):
            def hook(module, inputs, output):
                visual_grounding_features[name] = output.detach()
            return hook

        if not args.ablate_rgb:
            visual_grounding_hooks.append(
                trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(
                    cache_visual_feature("rgb")
                )
            )
        if not args.ablate_depth:
            visual_grounding_hooks.append(
                trainer.policy.net.depth_encoder.visual_encoder.register_forward_hook(
                    cache_visual_feature("depth")
                )
            )

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()
    gc.collect()


    #
    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            rgb_frames = [[] for _ in range(train_env.batch_size)]

            episodes = [[] for _ in range(train_env.batch_size)]
            action_histories = [[] for _ in range(train_env.batch_size)]
            action_guard = RepeatActionGuard(
                min_stop_step=args.EVAL_min_stop_step,
                max_vertical_repeat=args.EVAL_max_vertical_repeat,
                max_turn_repeat=args.EVAL_max_turn_repeat,
                max_action_repeat=args.EVAL_max_action_repeat,
                inverse_action_cooldown=args.EVAL_inverse_action_cooldown,
                repeat_recovery_action=args.EVAL_repeat_recovery_action,
            )
            depth_safety_guards = [
                DepthSafetyGuard(
                    threshold=args.EVAL_depth_safety_threshold,
                    danger_fraction=args.EVAL_depth_safety_danger_fraction,
                    turn_hold=args.EVAL_depth_safety_turn_hold,
                    climb_after=args.EVAL_depth_safety_climb_after,
                    climb_steps=args.EVAL_depth_safety_climb_steps,
                )
                for _ in range(train_env.batch_size)
            ]
            segment_states = None
            segment_grounding_controllers = None
            if args.EVAL_segment_instruction:
                if train_env.batch_size != 1:
                    raise ValueError("segmented instruction evaluation requires batchSize=1")
                segment_states = []
                for batch_index, episode in enumerate(train_env.batch):
                    segment_state = SegmentInstructionState(
                        episode["instruction"]["instruction_text"],
                        min_steps=args.EVAL_segment_min_steps,
                        max_steps=args.EVAL_segment_max_steps,
                        use_cues=args.EVAL_segment_use_cues,
                    )
                    segment_states.append(segment_state)
                    _set_eval_instruction_segment(
                        train_env, batch_index, segment_state.text, tok
                    )
                    logger.warning(
                        "SEGMENT_INIT episode=%s count=%d segment=1 bounds=%d:%d text=%s",
                        episode["episode_id"],
                        len(segment_state.segments),
                        segment_state.min_steps,
                        segment_state.max_steps,
                        segment_state.text,
                    )
                if args.EVAL_learned_segment_grounding:
                    if not args.EVAL_segment_grounding_ridge:
                        raise ValueError(
                            "learned segment grounding requires --EVAL_segment_grounding_ridge"
                        )
                    segment_grounding_controllers = [
                        SegmentGroundingController(
                            model_path=args.EVAL_segment_grounding_ridge,
                            final_model_path=args.EVAL_segment_grounding_final_ridge,
                            instruction=episode["instruction"]["instruction_text"],
                            scene_id=episode.get("scene_id"),
                            target_radius=args.EVAL_segment_grounding_radius,
                            length_model_path=args.EVAL_segment_grounding_length_model,
                            max_local_hops=args.EVAL_segment_grounding_max_local_hops,
                            final_fallback_model_path=args.EVAL_segment_grounding_final_fallback_ridge,
                            final_fallback_min_displacement=args.EVAL_segment_grounding_final_fallback_min_displacement,
                            final_ood_fallback_model_path=args.EVAL_segment_grounding_final_ood_fallback_ridge,
                            final_ood_min_score=args.EVAL_segment_grounding_final_ood_min_score,
                            intermediate_full_model_path=args.EVAL_segment_grounding_intermediate_full_ridge,
                            intermediate_full_max_nearest_score=args.EVAL_segment_grounding_intermediate_full_max_nearest_score,
                            intermediate_full_max_spread=args.EVAL_segment_grounding_intermediate_full_max_spread,
                        )
                        for episode in train_env.batch
                    ]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)
            if segment_grounding_controllers is not None:
                for batch_index, controller in enumerate(segment_grounding_controllers):
                    pose = train_env.sim_states[batch_index].pose
                    position = [
                        float(pose.position.x_val),
                        float(pose.position.y_val),
                        float(pose.position.z_val),
                    ]
                    yaw = airsim.to_eularian_angles(pose.orientation)[2]
                    target = controller.start_segment(0, position, yaw)
                    logger.warning(
                        "SEGMENT_GROUNDING_INIT episode=%s position=%s target=%s full=%s",
                        train_env.batch[batch_index]["episode_id"],
                        position,
                        target,
                        controller.used_intermediate_full_model,
                    )

            ended = False

            for t in range(int(args.maxAction)):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(checkpoint_index, idx, t, end_iter, not_done_masks.cpu().numpy().reshape((-1,)).tolist()))

                actions, rnn_states, action_info = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                    step=t,
                    return_info=True,
                )
                model_actions = actions.clone()
                if segment_grounding_controllers is not None:
                    for batch_index, controller in enumerate(
                        segment_grounding_controllers
                    ):
                        if controller.target is not None:
                            continue
                        pose = train_env.sim_states[batch_index].pose
                        position = [
                            float(pose.position.x_val),
                            float(pose.position.y_val),
                            float(pose.position.z_val),
                        ]
                        yaw = airsim.to_eularian_angles(pose.orientation)[2]
                        rgb_feature = visual_grounding_features["rgb"]
                        depth_feature = visual_grounding_features["depth"]
                        target = controller.start_segment(
                            segment_states[batch_index].index,
                            position,
                            yaw,
                            rgb_features=(
                                rgb_feature[batch_index]
                                if rgb_feature is not None
                                else None
                            ),
                            depth_features=(
                                depth_feature[batch_index]
                                if depth_feature is not None
                                else None
                            ),
                        )
                        if target is not None:
                            logger.warning(
                                "SEGMENT_GROUNDING_VISUAL_INIT episode=%s segment=%d position=%s target=%s",
                                train_env.batch[batch_index]["episode_id"],
                                segment_states[batch_index].index + 1,
                                position,
                                target,
                            )
                model_confidences = torch.softmax(
                    action_info["action_logits"], dim=1
                ).max(dim=1).values
                while True:
                    actions = action_guard.select(
                        action_info["action_logits"], action_histories, t, actions
                    )
                    if segment_states is None:
                        break
                    segment_state = segment_states[0]
                    proposed_action = int(actions[0, 0].item())
                    model_proposed_action = int(model_actions[0, 0].item())
                    model_confidence = float(model_confidences[0].item())
                    grounding_controller = (
                        segment_grounding_controllers[0]
                        if segment_grounding_controllers is not None
                        else None
                    )
                    grounded_decision = None
                    if grounding_controller is not None and grounding_controller.enabled:
                        pose = train_env.sim_states[0].pose
                        position = [
                            float(pose.position.x_val),
                            float(pose.position.y_val),
                            float(pose.position.z_val),
                        ]
                        yaw = airsim.to_eularian_angles(pose.orientation)[2]
                        current_grounding_distance = grounding_controller.observe(position)
                        final_goal_grounding = bool(
                            segment_state.is_last
                            and grounding_controller.final_target_is_goal
                        )
                        grounding_target_reached = bool(
                            grounding_controller.best_distance
                            <= (
                                float(args.EVAL_segment_grounding_final_radius)
                                if final_goal_grounding
                                else float(args.EVAL_segment_grounding_radius)
                            )
                        )
                        recovery_limit = int(
                            args.EVAL_segment_grounding_final_max_recovery
                            if final_goal_grounding
                            else args.EVAL_segment_grounding_max_recovery
                        )
                        grounding_min_steps = int(
                            args.EVAL_segment_grounding_final_min_steps
                            if final_goal_grounding
                            else args.EVAL_segment_grounding_min_steps
                        )
                        near_boundary = segment_state.steps >= max(
                            grounding_min_steps,
                            int(args.EVAL_segment_grounding_max_steps)
                            - int(args.EVAL_segment_grounding_boundary_window),
                        )
                        if (
                            not segment_state.is_last
                            and segment_state.steps
                            >= grounding_min_steps
                            and grounding_target_reached
                        ):
                            next_local_target = grounding_controller.complete_hop_and_retarget(
                                position,
                                yaw,
                                rgb_features=(
                                    visual_grounding_features["rgb"][0]
                                    if visual_grounding_features["rgb"] is not None
                                    else None
                                ),
                                depth_features=(
                                    visual_grounding_features["depth"][0]
                                    if visual_grounding_features["depth"] is not None
                                    else None
                                ),
                            )
                            if next_local_target is None:
                                grounded_decision = "advance_grounded_target"
                            else:
                                logger.warning(
                                    "SEGMENT_GROUNDING_RECENTER episode=%s segment=%d hop=%d/%d position=%s target=%s",
                                    train_env.batch[0]["episode_id"],
                                    segment_state.index + 1,
                                    grounding_controller.completed_hops + 1,
                                    grounding_controller.expected_hops,
                                    position,
                                    next_local_target,
                                )
                                recenter_action = grounding_controller.recovery_action(
                                    position,
                                    yaw,
                                    {
                                        "forward": AirsimActions.MOVE_FORWARD,
                                        "left": AirsimActions.TURN_LEFT,
                                        "right": AirsimActions.TURN_RIGHT,
                                        "up": AirsimActions.GO_UP,
                                        "down": AirsimActions.GO_DOWN,
                                    },
                                )
                                if recenter_action is not None:
                                    actions[0, 0] = int(recenter_action)
                                grounded_decision = "grounded_recovery"
                        elif (
                            final_goal_grounding
                            and segment_state.steps
                            >= grounding_min_steps
                            and grounding_target_reached
                        ):
                            actions[0, 0] = SegmentInstructionState.STOP
                            logger.warning(
                                "SEGMENT_GROUNDING_FINAL_STOP episode=%s step=%d distance=%.3f best=%.3f",
                                train_env.batch[0]["episode_id"],
                                segment_state.steps,
                                current_grounding_distance,
                                grounding_controller.best_distance,
                            )
                            grounded_decision = "grounded_final_stop"
                        elif (
                            (not segment_state.is_last or final_goal_grounding)
                            and not grounding_target_reached
                            and grounding_controller.recovery_steps
                            >= recovery_limit
                        ):
                            if final_goal_grounding:
                                actions[0, 0] = SegmentInstructionState.STOP
                                grounded_decision = "grounded_final_timeout_stop"
                            else:
                                grounded_decision = "advance_grounded_timeout"
                        elif (
                            (not segment_state.is_last or final_goal_grounding)
                            and not grounding_target_reached
                            and segment_state.steps >= grounding_min_steps
                            and (
                                model_proposed_action == SegmentInstructionState.STOP
                                or segment_state.steps
                                >= int(args.EVAL_segment_grounding_max_steps)
                                or (
                                    near_boundary
                                    and model_confidence
                                    < float(args.EVAL_segment_grounding_low_confidence)
                                )
                            )
                        ):
                            recovery_action = grounding_controller.recovery_action(
                                position,
                                yaw,
                                {
                                    "forward": AirsimActions.MOVE_FORWARD,
                                    "left": AirsimActions.TURN_LEFT,
                                    "right": AirsimActions.TURN_RIGHT,
                                    "up": AirsimActions.GO_UP,
                                    "down": AirsimActions.GO_DOWN,
                                },
                            )
                            if recovery_action is not None:
                                actions[0, 0] = int(recovery_action)
                                logger.info(
                                    "SEGMENT_GROUNDING_RECOVERY episode=%s segment=%d step=%d distance=%.3f best=%.3f confidence=%.3f action=%d",
                                    train_env.batch[0]["episode_id"],
                                    segment_state.index + 1,
                                    segment_state.steps,
                                    current_grounding_distance,
                                    grounding_controller.best_distance,
                                    model_confidence,
                                    recovery_action,
                                )
                                grounded_decision = "grounded_recovery"
                    if grounded_decision in {
                        "grounded_recovery",
                        "grounded_final_stop",
                        "grounded_final_timeout_stop",
                    }:
                        break
                    expected_action = segment_state.expected_action
                    if grounded_decision is not None:
                        decision = grounded_decision
                    elif expected_action is not None:
                        if proposed_action != expected_action:
                            logger.info(
                                "SEGMENT_CUE_OVERRIDE episode=%s segment=%d step=%d model=%d expected=%d",
                                train_env.batch[0]["episode_id"],
                                segment_state.index + 1,
                                segment_state.steps,
                                proposed_action,
                                expected_action,
                            )
                        actions[0, 0] = expected_action
                        decision = "act"
                    else:
                        decision = segment_state.decision(proposed_action)
                    if decision.startswith("advance_"):
                        segment_state.advance()
                        if grounding_controller is not None:
                            pose = train_env.sim_states[0].pose
                            position = [
                                float(pose.position.x_val),
                                float(pose.position.y_val),
                                float(pose.position.z_val),
                            ]
                            yaw = airsim.to_eularian_angles(pose.orientation)[2]
                            next_grounding_target = grounding_controller.start_segment(
                                segment_state.index,
                                position,
                                yaw,
                                rgb_features=(
                                    visual_grounding_features["rgb"][0]
                                    if visual_grounding_features["rgb"] is not None
                                    else None
                                ),
                                depth_features=(
                                    visual_grounding_features["depth"][0]
                                    if visual_grounding_features["depth"] is not None
                                    else None
                                ),
                            )
                            logger.warning(
                                "SEGMENT_GROUNDING_TARGET episode=%s segment=%d position=%s target=%s full=%s",
                                train_env.batch[0]["episode_id"],
                                segment_state.index + 1,
                                position,
                                next_grounding_target,
                                grounding_controller.used_intermediate_full_model,
                            )
                        segment_tokens = _set_eval_instruction_segment(
                            train_env, 0, segment_state.text, tok
                        )
                        batch["instruction"][0].copy_(
                            torch.as_tensor(
                                segment_tokens,
                                device=trainer.device,
                                dtype=batch["instruction"].dtype,
                            )
                        )
                        rnn_states[0].zero_()
                        prev_actions[0].zero_()
                        not_done_masks[0].zero_()
                        action_histories[0].clear()
                        logger.warning(
                            "SEGMENT_ADVANCE episode=%s segment=%d/%d reason=%s bounds=%d:%d text=%s",
                            train_env.batch[0]["episode_id"],
                            segment_state.index + 1,
                            len(segment_state.segments),
                            decision,
                            segment_state.min_steps,
                            segment_state.max_steps,
                            segment_state.text,
                        )
                        actions, rnn_states, action_info = trainer.policy.act(
                            batch,
                            rnn_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=True,
                            step=t,
                            return_info=True,
                        )
                        model_actions = actions.clone()
                        model_confidences = torch.softmax(
                            action_info["action_logits"], dim=1
                        ).max(dim=1).values
                        continue
                    if decision == "block_early_stop":
                        non_stop_logits = action_info["action_logits"].clone()
                        non_stop_logits[0, SegmentInstructionState.STOP] = torch.finfo(
                            non_stop_logits.dtype
                        ).min
                        actions = non_stop_logits.argmax(dim=1, keepdim=True)
                        actions = action_guard.select(
                            non_stop_logits, action_histories, t, actions
                        )
                    break

                if args.EVAL_depth_safety:
                    for batch_index, safety_guard in enumerate(depth_safety_guards):
                        proposed = int(actions[batch_index, 0].item())
                        safe_action, safety_detail = safety_guard.select(
                            proposed,
                            observations[batch_index].get("depth"),
                            {
                                "forward": AirsimActions.MOVE_FORWARD,
                                "left": AirsimActions.TURN_LEFT,
                                "right": AirsimActions.TURN_RIGHT,
                                "up": AirsimActions.GO_UP,
                            },
                        )
                        if safe_action != proposed:
                            actions[batch_index, 0] = int(safe_action)
                            logger.warning(
                                "DEPTH_SAFETY_OVERRIDE episode=%s step=%d model=%d safe=%d detail=%s",
                                train_env.batch[batch_index]["episode_id"],
                                t,
                                proposed,
                                safe_action,
                                safety_detail,
                            )
                prev_actions.copy_(actions)

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()]
                for history, action in zip(action_histories, actions):
                    history.append(int(action))
                if segment_states is not None:
                    segment_states[0].record_action()
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

                # for tttt in range(len(train_env.batch)):
                #     train_env.threading_lock_lmdb_features_txn.acquire()
                #     train_env.lmdb_features_txn.put(
                #         str('{}_{}_{}'.format(infos[tttt]['episode_id'], t, 'exp_1')).encode(),
                #         msgpack_numpy.packb(
                #             observations[tttt], use_bin_type=True
                #         ),
                #     )
                #     train_env.lmdb_features_txn.commit()
                #     train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                #     train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                #     train_env.threading_lock_lmdb_features_txn.release()
                #     logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                # reset envs and observations if necessary
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i])
                        frame = append_text_to_image(
                            frame, train_env.batch[i]['instruction']['instruction_text']
                        )
                        rgb_frames[i].append(frame)

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index))):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'success: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))

    # end
    pbar.close()


    #
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    #
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()

    if args.run_type == 'collect':
        collect_data()
    elif args.run_type == 'train':
        train_vlnce()
    elif args.run_type == 'eval':
        eval_vlnce()
    else:
        raise NotImplementedError

