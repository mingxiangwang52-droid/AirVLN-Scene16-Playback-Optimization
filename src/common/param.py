import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        project_prefix = Path(str(os.getcwd())).parent.resolve()
        self.parser.add_argument('--project_prefix', type=str, default=str(project_prefix), help="project path")

        self.parser.add_argument('--run_type', type=str, default="train", help="run_type in [collect, train, eval]")
        self.parser.add_argument('--policy_type', type=str, default="seq2seq", help="policy_type in [seq2seq, cma]")
        self.parser.add_argument('--collect_type', type=str, default="TF", help="seq2seq in [TF, dagger, SF]")
        self.parser.add_argument('--name', type=str, default='default', help='experiment name')

        self.parser.add_argument('--maxInput', type=int, default=300, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=500, help='max action sequence')

        self.parser.add_argument("--dagger_it", type=int, default=1)
        self.parser.add_argument("--epochs", type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.00025, help="learning rate")
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument("--trainer_gpu_device", type=int, default=0, help='GPU')
        self.parser.add_argument('--init_checkpoint', type=str, default='')
        self.parser.add_argument('--overfit_holdout_fraction', type=float, default=0.0)
        self.parser.add_argument('--overfit_holdout_seed', type=int, default=20260719)
        self.parser.add_argument('--replay_feature_dir', type=str, default='')
        self.parser.add_argument('--replay_primary_per_replay', type=int, default=0)

        self.parser.add_argument('--Image_Height_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Width_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Height_DEPTH', type=int, default=256)
        self.parser.add_argument('--Image_Width_DEPTH', type=int, default=256)

        self.parser.add_argument('--inflection_weight_coef', type=float, default=1.9)
        self.parser.add_argument('--stop_weight_coef', type=float, default=1.0)
        self.parser.add_argument('--vertical_repeat_weight', type=float, default=1.0)
        self.parser.add_argument('--vertical_repeat_grace_steps', type=int, default=0)
        self.parser.add_argument('--prev_action_token_dropout', type=float, default=0.0)
        self.parser.add_argument(
            '--freeze_feature_encoders',
            action='store_true',
            help='Freeze RGB, depth, and instruction encoders during fine-tuning.',
        )
        self.parser.add_argument(
            '--reset_optimizer_on_init',
            action='store_true',
            help='Load checkpoint weights without restoring its optimizer state.',
        )
        self.parser.add_argument('--segment_train_instructions', action='store_true')
        self.parser.add_argument('--segment_train_data_path', type=str, default='')
        self.parser.add_argument(
            '--segment_train_max_segments_per_trajectory', type=int, default=0
        )
        self.parser.add_argument('--train_source_limit', type=int, default=0)
        self.parser.add_argument('--collect_beta', type=float, default=1.0)
        self.parser.add_argument(
            '--collect_reference_path_teacher',
            action='store_true',
            help='Use episode reference_path for DAgger labels when navigation graphs are unavailable.',
        )
        self.parser.add_argument(
            '--collect_prevent_early_stop',
            action='store_true',
            help='Execute the teacher recovery action when the sampled policy stops before the teacher.',
        )

        self.parser.add_argument('--nav_graph_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/nav_graph_10'), help="nav_graph path")
        self.parser.add_argument('--token_dict_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/token_dict_10'), help="token_dict path")
        self.parser.add_argument('--vertices_path', type=str, default=str(project_prefix / 'DATA/data/disceret/scene_meshes'))
        self.parser.add_argument('--dagger_mode_load_scene', nargs='+', default=[])
        self.parser.add_argument('--dagger_update_size', type=int, default=8000)
        self.parser.add_argument('--dagger_mode', type=str, default="end", help='dagger mode in [end middle nearest]')
        self.parser.add_argument('--dagger_p', type=float, default=1.0, help='dagger p')

        self.parser.add_argument('--TF_mode_load_scene', nargs='+', default=[])
        self.parser.add_argument('--TRAIN_SPLIT', type=str, default='train', help='training split json name under DATA/data/aerialvln')

        self.parser.add_argument('--ablate_instruction', action="store_true")
        self.parser.add_argument('--ablate_rgb', action="store_true")
        self.parser.add_argument('--ablate_depth', action="store_true")
        self.parser.add_argument('--SEQ2SEQ_use_prev_action', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_use', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_alpha', type=float, default=1.0)
        self.parser.add_argument('--SEGMENT_ALIGN_use', action="store_true")
        self.parser.add_argument('--SEGMENT_ALIGN_alpha', type=float, default=0.05)
        self.parser.add_argument('--SEGMENT_ALIGN_temperature', type=float, default=0.07)
        self.parser.add_argument('--SEGMENT_ALIGN_symmetric', action="store_true")
        self.parser.add_argument('--SEGMENT_ALIGN_mode', type=str, default='contrastive')
        self.parser.add_argument('--SEGMENT_ALIGN_start_epoch', type=int, default=2)
        self.parser.add_argument('--SEGMENT_ALIGN_warmup_epochs', type=int, default=4)
        self.parser.add_argument('--SEGMENT_ALIGN_min_scale', type=float, default=0.0)
        self.parser.add_argument('--SEGMENT_ALIGN_boundary_only', action="store_true")
        self.parser.add_argument('--SEGMENT_ALIGN_boundary_start_window', type=float, default=0.2)
        self.parser.add_argument('--SEGMENT_ALIGN_boundary_end_window', type=float, default=0.2)

        self.parser.add_argument('--EVAL_CKPT_PATH_DIR', type=str)
        self.parser.add_argument('--EVAL_DATASET', type=str, default="val_unseen")
        self.parser.add_argument("--EVAL_NUM", type=int, default=-1)
        self.parser.add_argument('--EVAL_GENERATE_VIDEO', action="store_true")
        self.parser.add_argument('--EVAL_min_stop_step', type=int, default=0)
        self.parser.add_argument('--EVAL_max_vertical_repeat', type=int, default=0)
        self.parser.add_argument('--EVAL_max_turn_repeat', type=int, default=0)
        self.parser.add_argument('--EVAL_max_action_repeat', type=int, default=0)
        self.parser.add_argument('--EVAL_inverse_action_cooldown', type=int, default=0)
        self.parser.add_argument('--EVAL_repeat_recovery_action', type=int, default=0)
        self.parser.add_argument('--EVAL_segment_instruction', action="store_true")
        self.parser.add_argument('--EVAL_segment_use_cues', action="store_true")
        self.parser.add_argument('--EVAL_segment_min_steps', type=int, default=3)
        self.parser.add_argument('--EVAL_segment_max_steps', type=int, default=32)
        self.parser.add_argument('--EVAL_learned_segment_grounding', action="store_true")
        self.parser.add_argument('--EVAL_segment_grounding_ridge', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_final_ridge', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_length_model', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_max_local_hops', type=int, default=6)
        self.parser.add_argument('--EVAL_segment_grounding_final_fallback_ridge', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_final_fallback_min_displacement', type=float, default=0.0)
        self.parser.add_argument('--EVAL_segment_grounding_final_ood_fallback_ridge', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_final_ood_min_score', type=float, default=0.0)
        self.parser.add_argument('--EVAL_segment_grounding_intermediate_full_ridge', type=str, default='')
        self.parser.add_argument('--EVAL_segment_grounding_intermediate_full_max_nearest_score', type=float, default=0.0)
        self.parser.add_argument('--EVAL_segment_grounding_intermediate_full_max_spread', type=float, default=0.0)
        self.parser.add_argument('--EVAL_segment_grounding_radius', type=float, default=8.0)
        self.parser.add_argument('--EVAL_segment_grounding_final_radius', type=float, default=3.0)
        self.parser.add_argument('--EVAL_segment_grounding_min_steps', type=int, default=3)
        self.parser.add_argument('--EVAL_segment_grounding_final_min_steps', type=int, default=3)
        self.parser.add_argument('--EVAL_segment_grounding_max_steps', type=int, default=12)
        self.parser.add_argument('--EVAL_segment_grounding_boundary_window', type=int, default=6)
        self.parser.add_argument('--EVAL_segment_grounding_low_confidence', type=float, default=0.35)
        self.parser.add_argument('--EVAL_segment_grounding_max_recovery', type=int, default=12)
        self.parser.add_argument('--EVAL_segment_grounding_final_max_recovery', type=int, default=40)
        self.parser.add_argument('--EVAL_depth_safety', action="store_true")
        self.parser.add_argument('--EVAL_depth_safety_threshold', type=float, default=0.07)
        self.parser.add_argument('--EVAL_depth_safety_danger_fraction', type=float, default=0.08)
        self.parser.add_argument('--EVAL_depth_safety_turn_hold', type=int, default=3)
        self.parser.add_argument('--EVAL_depth_safety_climb_after', type=int, default=3)
        self.parser.add_argument('--EVAL_depth_safety_climb_steps', type=int, default=2)

        self.parser.add_argument('--rgb_encoder_use_place365', action="store_true")
        self.parser.add_argument('--tokenizer_use_bert', action="store_true")

        self.parser.add_argument("--simulator_tool_port", type=int, default=30000, help="simulator_tool port")
        self.parser.add_argument("--DDP_MASTER_PORT", type=int, default=20000, help="DDP MASTER_PORT")

        self.parser.add_argument("--continue_start_from_dagger_it", type=int)
        self.parser.add_argument("--continue_start_from_checkpoint_path", type=str)

        self.parser.add_argument('--vlnbert', action="store_true", default="prevalent")
        self.parser.add_argument('--featdropout', action="store_true", default=0.4)
        self.parser.add_argument('--action_feature', action="store_true", default=32)

        self.args = self.parser.parse_args()


param = Param()
args = param.args

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/DATA/output/{}/{}/logs/{}_{}.log'.format(args.project_prefix, args.name, args.run_type, args.name, args.make_dir_time)


# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval'], 'run_type error'
# args.policy_type = 'seq2seq'
assert args.policy_type in ['seq2seq', 'cma'], 'policy_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'


args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]


args.TRAIN_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038


default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()

