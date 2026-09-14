import os
from typing import Dict

import torch
import torch.nn.functional as F
from gym import Space

from Model.seq2seq_policy import Seq2SeqPolicy
from Model.cma_policy import CMAPolicy
from utils.logger import logger
from src.common.param import args
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN


class VLNCETrainer:
    #
    def __init__(
        self,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        ckpt_path=None,
    ):
        self.start_epoch = 0
        self.step_id = 0

        if not args.DistributedDataParallel:
            self.device = (
                torch.device("cuda", args.trainer_gpu_device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        model_config = CN.clone()
        if args.policy_type == 'seq2seq':
            self.policy = Seq2SeqPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        elif args.policy_type == 'cma':
            self.policy = CMAPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        elif args.policy_type == 'hcm':
            self.policy = HCMPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        elif args.policy_type == 'unet':
            self.policy = UNetPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        elif args.policy_type == 'vlnbert':
            self.policy = VLNBertPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        else:
            raise NotImplementedError

        self.policy.to(self.device)

        if bool(getattr(args, "freeze_feature_encoders", False)):
            frozen_modules = []
            for module_name in (
                "instruction_encoder",
                "rgb_encoder",
                "depth_encoder",
            ):
                module = getattr(self.policy.net, module_name, None)
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad = False
                frozen_modules.append(module_name)
            logger.info("Frozen feature encoders: %s", ", ".join(frozen_modules))

        trainable_parameters = [
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            trainable_parameters, lr=args.lr
        )

        if load_from_ckpt:
            assert os.path.isfile(ckpt_path), 'ckpt_path error'
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            load_result = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing checkpoint keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")
            if not bool(getattr(args, "reset_optimizer_on_init", False)):
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optimizer"])
                except Exception as exc:
                    logger.warning(f"Skipping optimizer state load due to mismatch: {exc}")
            else:
                logger.info("Reset optimizer state after loading checkpoint weights")
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        if args.DistributedDataParallel:
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    #
    def save_checkpoint(self, file_name, dagger_it, epoch) -> None:
        """
        Save checkpoint with specified name.
        :param file_name: file name for checkpoint
        :param epoch: epoch
        :return: None
        """
        checkpoint = {
            "state_dict": self.policy.module.state_dict() if args.DistributedDataParallel else self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "config": str(args),
            'dagger_it': dagger_it,
            'epoch': epoch,
        }

        from pathlib import Path
        checkpoint_folder = Path(args.project_prefix) / 'DATA/output/{}/train/checkpoint/{}'.format(args.name, args.make_dir_time)
        if not os.path.exists(str(checkpoint_folder)):
            os.makedirs(str(checkpoint_folder), exist_ok=True)

        torch.save(
            checkpoint, str(checkpoint_folder / file_name)
        )

    #
    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    #
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
        backward: bool = True,
    ):
        T, N = corrected_actions.size()

        if args.policy_type in ['seq2seq', 'cma']:
            if not args.DistributedDataParallel:
                recurrent_hidden_states = torch.zeros(
                    N,
                    self.policy.net.num_recurrent_layers,
                    self.policy.net.state_encoder.hidden_size,
                    device=self.device,
                )
            else:
                recurrent_hidden_states = torch.zeros(
                    N,
                    self.policy.module.net.num_recurrent_layers,
                    self.policy.module.net.state_encoder.hidden_size,
                    device=self.device,
                )
        else:
            raise NotImplementedError

        AuxLosses.clear()

        if not args.DistributedDataParallel:
            distribution = self.policy.build_distribution(
                observations, recurrent_hidden_states, prev_actions, not_done_masks
            )
        else:
            distribution = self.policy.module.build_distribution(
                observations, recurrent_hidden_states, prev_actions, not_done_masks
            )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        valid_actions = (corrected_actions >= 0) & (
            corrected_actions < logits.size(-1)
        )
        if not bool(valid_actions.all()):
            invalid_count = int((~valid_actions).sum().item())
            logger.warning(
                "Ignoring %d invalid teacher action labels outside [0, %d)",
                invalid_count,
                logits.size(-1),
            )
            corrected_actions = corrected_actions.masked_fill(~valid_actions, 0)
            weights = weights * valid_actions.to(dtype=weights.dtype)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = (
            (weights * action_loss).sum(0)
            / weights.sum(0).clamp_min(1.0)
        ).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        if backward:
            loss.backward()

            if step_grad:
                self.optimizer.step()
                self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

