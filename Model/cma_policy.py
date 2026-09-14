import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from Model.policy import ILPolicy
from Model.encoders.instruction_encoder import InstructionEncoder, InstructionBertEncoder
from Model.encoders.resnet_encoders import TorchVisionResNet50, TorchVisionResNet50Place365, VlnResnetDepthEncoder
from Model.encoders.rnn_state_encoder import build_rnn_state_encoder
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN

from src.common.param import args


class CMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space,
                num_actions=action_space.n,
                out_model_config=out_model_config,
                device=device,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, observation_space: Space, action_space: Space, out_model_config=None,
        device=torch.device("cpu"),
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            out_model_config=out_model_config,
            device=device,
        )


class CMANet(nn.Module):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    CMA state encoder
    """

    def __init__(
        self, observation_space: Space, num_actions, out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device

        model_config = CN.clone()
        model_config.STATE_ENCODER_hidden_size = 512
        model_config.STATE_ENCODER_rnn_type = 'GRU'
        model_config.PROGRESS_MONITOR_use = args.PROGRESS_MONITOR_use
        model_config.PROGRESS_MONITOR_alpha = args.PROGRESS_MONITOR_alpha
        self.model_config = model_config

        # Init the instruction encoder 1
        if args.tokenizer_use_bert:
            self.instruction_encoder = InstructionBertEncoder()
        else:
            self.instruction_encoder = InstructionEncoder()

        # Init the depth encoder 2
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
        )

        # Init the RGB encoder 3
        if args.rgb_encoder_use_place365:
            self.rgb_encoder = TorchVisionResNet50Place365(
                observation_space, device,
            )
        else:
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, device,
            )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER_hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                self.rgb_encoder.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                self.depth_encoder.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = self.depth_encoder.output_size
        rnn_input_size += self.rgb_encoder.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER_hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER_hidden_size
            + self.rgb_encoder.output_size
            + self.depth_encoder.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + self.rgb_encoder.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + self.depth_encoder.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )
        self._output_size = model_config.STATE_ENCODER_hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)
        self.segment_align_enabled = bool(args.SEGMENT_ALIGN_use)
        self.latest_alignment_scores = None
        self.latest_alignment_confidence = None
        if self.segment_align_enabled:
            self.segment_visual_proj = nn.Linear(
                hidden_size + self.rgb_encoder.output_size + self.depth_encoder.output_size,
                hidden_size,
            )
            self.segment_text_proj = nn.Linear(
                self.instruction_encoder.output_size,
                hidden_size,
            )

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        if self.model_config.PROGRESS_MONITOR_use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def _pool_text_instruction(self, instruction_embedding):
        token_mask = (instruction_embedding == 0.0).all(dim=1)
        valid = (~token_mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (instruction_embedding * valid.unsqueeze(1)).sum(dim=2) / denom
        return pooled

    def _segment_align_alpha(self):
        base_alpha = float(args.SEGMENT_ALIGN_alpha)
        if base_alpha <= 0.0:
            return 0.0

        current_epoch = int(getattr(args, "segment_align_current_epoch", 0))
        start_epoch = max(int(args.SEGMENT_ALIGN_start_epoch), 0)
        warmup_epochs = max(int(args.SEGMENT_ALIGN_warmup_epochs), 0)
        min_scale = float(args.SEGMENT_ALIGN_min_scale)

        if current_epoch < start_epoch:
            return 0.0

        if warmup_epochs == 0:
            return base_alpha

        warmup_progress = min(current_epoch - start_epoch + 1, warmup_epochs)
        warmup_scale = warmup_progress / float(warmup_epochs)
        scaled_alpha = base_alpha * max(min_scale, warmup_scale)
        return min(base_alpha, scaled_alpha)

    def _focus_segment_alignment_loss(self, align_loss, observations):
        if not bool(getattr(args, "SEGMENT_ALIGN_boundary_only", False)):
            return align_loss

        progress = observations.get("progress")
        if progress is None:
            return align_loss

        progress = progress.reshape(-1)
        if progress.numel() != align_loss.numel():
            return align_loss

        raw_start_window = float(args.SEGMENT_ALIGN_boundary_start_window)
        raw_end_window = float(args.SEGMENT_ALIGN_boundary_end_window)
        start_mask = torch.zeros_like(progress, dtype=torch.bool)
        end_mask = torch.zeros_like(progress, dtype=torch.bool)

        if raw_start_window >= 0.0:
            start_window = min(raw_start_window, 1.0)
            start_mask = progress <= start_window

        if raw_end_window >= 0.0:
            end_window = min(raw_end_window, 1.0)
            end_mask = progress >= (1.0 - end_window)

        boundary_mask = start_mask | end_mask

        if not boundary_mask.any():
            return torch.zeros_like(align_loss)

        boundary_count = boundary_mask.float().sum().clamp_min(1.0)
        rescale = float(boundary_mask.numel()) / boundary_count
        focused_loss = torch.zeros_like(align_loss)
        focused_loss[boundary_mask] = align_loss[boundary_mask] * rescale
        return focused_loss

    def _trajectory_segment_alignment_loss(
        self,
        visual_features,
        text_features,
        observations,
        trajectory_batch_size,
    ):
        valid = observations.get("trajectory_valid")
        batch_size = int(trajectory_batch_size)
        if valid is None or batch_size <= 0 or visual_features.size(0) % batch_size != 0:
            return None

        time_steps = visual_features.size(0) // batch_size
        valid = valid.reshape(time_steps, batch_size) > 0.5
        if not valid.any():
            return None

        active = valid.clone()
        if bool(getattr(args, "SEGMENT_ALIGN_boundary_only", False)):
            progress = observations.get("progress")
            if progress is not None and progress.numel() == valid.numel():
                progress = progress.reshape(time_steps, batch_size)
                active = torch.zeros_like(valid)
                start_window = float(args.SEGMENT_ALIGN_boundary_start_window)
                end_window = float(args.SEGMENT_ALIGN_boundary_end_window)
                if start_window >= 0.0:
                    active |= progress <= min(start_window, 1.0)
                if end_window >= 0.0:
                    active |= progress >= (1.0 - min(end_window, 1.0))
                active &= valid
                empty = ~active.any(dim=0)
                if empty.any():
                    active[:, empty] = valid[:, empty]

        visual = visual_features.reshape(time_steps, batch_size, -1)
        text = text_features.reshape(time_steps, batch_size, -1)
        active_count = active.float().sum(dim=0).clamp_min(1.0)
        valid_count = valid.float().sum(dim=0).clamp_min(1.0)
        visual_pooled = (visual * active.unsqueeze(2)).sum(dim=0) / active_count.unsqueeze(1)
        text_pooled = (text * valid.unsqueeze(2)).sum(dim=0) / valid_count.unsqueeze(1)

        visual_embed = F.normalize(self.segment_visual_proj(visual_pooled), p=2, dim=1)
        text_embed = F.normalize(self.segment_text_proj(text_pooled), p=2, dim=1)
        similarity = torch.matmul(visual_embed, text_embed.transpose(0, 1))
        diagonal = similarity.diag()
        align_mode = str(getattr(args, "SEGMENT_ALIGN_mode", "contrastive")).lower()
        if align_mode == "paired_cosine":
            trajectory_loss = 1.0 - diagonal
        else:
            temperature = max(float(args.SEGMENT_ALIGN_temperature), 1.0e-4)
            scaled_similarity = similarity / temperature
            target = torch.arange(batch_size, device=scaled_similarity.device)
            row_loss = F.cross_entropy(scaled_similarity, target, reduction="none")
            if bool(args.SEGMENT_ALIGN_symmetric):
                col_loss = F.cross_entropy(
                    scaled_similarity.transpose(0, 1), target, reduction="none"
                )
                trajectory_loss = 0.5 * (row_loss + col_loss)
            else:
                trajectory_loss = row_loss

        # AuxLosses averages over every valid time step. This scaling gives each
        # trajectory equal weight while applying alignment only at its boundaries.
        total_valid = valid.float().sum().clamp_min(1.0)
        scale = total_valid / (float(batch_size) * active_count)
        step_loss = torch.zeros_like(valid, dtype=trajectory_loss.dtype)
        step_loss[active] = (
            trajectory_loss.unsqueeze(0).expand(time_steps, -1)
            * scale.unsqueeze(0)
        )[active]
        step_scores = diagonal.unsqueeze(0).expand(time_steps, -1).reshape(-1)
        return step_loss.reshape(-1), step_scores

    def _register_segment_alignment_loss(
        self,
        visual_features,
        text_features,
        observations,
        trajectory_batch_size,
    ):
        trajectory_result = self._trajectory_segment_alignment_loss(
            visual_features,
            text_features,
            observations,
            trajectory_batch_size,
        )
        if trajectory_result is not None:
            align_loss, diagonal = trajectory_result
        else:
            visual_embed = F.normalize(self.segment_visual_proj(visual_features), p=2, dim=1)
            text_embed = F.normalize(self.segment_text_proj(text_features), p=2, dim=1)
            similarity = torch.matmul(visual_embed, text_embed.transpose(0, 1))
            diagonal = similarity.diag()
            align_mode = str(getattr(args, "SEGMENT_ALIGN_mode", "contrastive")).lower()
            if align_mode == "paired_cosine":
                align_loss = 1.0 - diagonal
            else:
                temperature = max(float(args.SEGMENT_ALIGN_temperature), 1.0e-4)
                scaled_similarity = similarity / temperature
                target = torch.arange(
                    scaled_similarity.size(0), device=scaled_similarity.device
                )
                row_loss = F.cross_entropy(
                    scaled_similarity, target, reduction="none"
                )
                if bool(args.SEGMENT_ALIGN_symmetric):
                    col_loss = F.cross_entropy(
                        scaled_similarity.transpose(0, 1),
                        target,
                        reduction="none",
                    )
                    align_loss = 0.5 * (row_loss + col_loss)
                else:
                    align_loss = row_loss
            align_loss = self._focus_segment_alignment_loss(align_loss, observations)
        self.latest_alignment_scores = diagonal.detach()
        self.latest_alignment_confidence = ((diagonal + 1.0) * 0.5).clamp(0.0, 1.0).detach()
        if AuxLosses.is_active():
            effective_alpha = self._segment_align_alpha()
            if effective_alpha <= 0.0:
                return
            AuxLosses.register_loss(
                "segment_alignment",
                align_loss,
                effective_alpha,
            )

    def forward(self, observations, rnn_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        prev_action_ids = ((prev_actions.float() + 1) * masks).long().view(-1)
        dropout_probability = min(
            max(float(getattr(args, "prev_action_token_dropout", 0.0)), 0.0),
            1.0,
        )
        if self.training and dropout_probability > 0.0:
            valid_previous_action = prev_action_ids != 0
            drop_previous_action = (
                torch.rand_like(prev_action_ids, dtype=torch.float32)
                < dropout_probability
            ) & valid_previous_action
            prev_action_ids = prev_action_ids.masked_fill(drop_previous_action, 0)
        prev_actions = self.prev_action_embedding(prev_action_ids)  # [BATCH x 32]

        if args.ablate_instruction:
            if self.instruction_encoder.config.final_state_only:
                instruction_embedding = torch.zeros(
                    size=(prev_actions.shape[0], self.instruction_encoder.output_size),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                instruction_embedding = torch.zeros(
                    size=(prev_actions.shape[0], self.instruction_encoder.output_size, (observations["instruction"] != 0.0).long().sum(dim=1).max().detach().cpu().numpy().tolist()),
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            instruction_embedding = self.instruction_encoder(observations)

        if args.ablate_depth:
            depth_embedding = torch.zeros(
                size=[prev_actions.shape[0]] + list(self.depth_encoder.output_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        if args.ablate_rgb:
            rgb_embedding = torch.zeros(
                size=[prev_actions.shape[0]]+list(self.rgb_encoder.output_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)
        text_global = self._pool_text_instruction(instruction_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        if self.segment_align_enabled:
            visual_global = torch.cat([state, rgb_in, depth_in], dim=1)
            self._register_segment_alignment_loss(
                visual_global,
                text_global,
                observations,
                rnn_states.size(0),
            )
        else:
            self.latest_alignment_scores = None
            self.latest_alignment_confidence = None

        if self.model_config.PROGRESS_MONITOR_use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_target = observations["progress"].reshape(-1)
            if progress_hat.size(0) != progress_target.numel():
                raise ValueError(
                    "progress monitor shape mismatch: "
                    f"prediction={tuple(progress_hat.shape)} "
                    f"target={tuple(observations['progress'].shape)}"
                )
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                progress_target,
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR_alpha,
            )

        return x, rnn_states_out

