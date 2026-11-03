"""RGBT-language reliability-aware dynamic sparse learning modules.

This file implements the three components used by the modified MLSTTrack backbone:
1) Modality-aware Language Routing (MLR)
2) Reliability-aware Dynamic Sparse Learning (RDSL)
3) Dynamic Multi-Level Prompt Perception (DMLP)

The implementation intentionally keeps the main iTPN token sequence length unchanged.
Multi-token language features are used only inside the sparse prompt branch, while one
modality-specific language token is kept in each RGB/TIR transformer stream.
"""

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAwareLanguageRouter(nn.Module):
    """Route language tokens into RGB-specific, TIR-specific and shared semantics.

    The routing decision is conditioned on the current RGB/TIR search context, so the
    same language description can be interpreted differently by the two sensors.
    A residual formulation is used to avoid suppressing the pretrained text features at
    the beginning of finetuning.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.lang_norm = nn.LayerNorm(dim)
        self.rgb_context = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.tir_context = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.route_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.rgb_out = nn.LayerNorm(dim)
        self.tir_out = nn.LayerNorm(dim)
        self.shared_out = nn.LayerNorm(dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        tir_tokens: torch.Tensor,
        rgb_visual: torch.Tensor,
        tir_visual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            rgb_tokens/tir_tokens: [B, K, C]
            rgb_visual/tir_visual: [B, N, C] or [B, 1, C]

        Returns:
            routed_rgb, routed_tir, routed_shared: [B, K, C]
            gates: [B, K, 3] ordered as RGB/TIR/shared.
        """
        if rgb_tokens.shape != tir_tokens.shape:
            raise ValueError(
                f"RGB/TIR language token shapes must match, got {rgb_tokens.shape} and {tir_tokens.shape}."
            )

        rgb_ctx = self.rgb_context(rgb_visual.mean(dim=1, keepdim=True))
        tir_ctx = self.tir_context(tir_visual.mean(dim=1, keepdim=True))
        rgb_ctx = rgb_ctx.expand(-1, rgb_tokens.size(1), -1)
        tir_ctx = tir_ctx.expand(-1, tir_tokens.size(1), -1)

        shared_base = 0.5 * (rgb_tokens + tir_tokens)
        route_input = torch.cat(
            [self.lang_norm(shared_base), rgb_ctx, tir_ctx], dim=-1
        )
        gates = F.softmax(self.route_mlp(route_input), dim=-1)
        gate_rgb = gates[..., 0:1]
        gate_tir = gates[..., 1:2]
        gate_shared = gates[..., 2:3]

        # Residual routing keeps the original modality-specific CLIP prompt stable,
        # while the learned gates amplify modality-matched/shared semantics.
        routed_rgb = self.rgb_out(
            rgb_tokens + gate_rgb * rgb_tokens + gate_shared * shared_base
        )
        routed_tir = self.tir_out(
            tir_tokens + gate_tir * tir_tokens + gate_shared * shared_base
        )
        routed_shared = self.shared_out(
            shared_base + gate_shared * shared_base
        )
        return routed_rgb, routed_tir, routed_shared, gates


class ReliabilityEstimator(nn.Module):
    """Estimate RGB/TIR reliability and reliability-conditioned sparsity mixtures."""

    def __init__(
        self,
        dim: int,
        spatial_ratios: Sequence[float],
        channel_ratios: Sequence[float],
        hidden_dim: int = 128,
        reliability_bias_strength: float = 2.0,
    ):
        super().__init__()
        self.spatial_ratios = tuple(float(v) for v in spatial_ratios)
        self.channel_ratios = tuple(float(v) for v in channel_ratios)
        self.reliability_bias_strength = float(reliability_bias_strength)

        in_dim = dim * 4  # RGB, TIR, shared language, temporal/history context
        self.context = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.reliability_head = nn.Linear(hidden_dim, 2)
        self.spatial_head = nn.Linear(hidden_dim, 2 * len(self.spatial_ratios))
        self.channel_head = nn.Linear(hidden_dim, 2 * len(self.channel_ratios))

    @staticmethod
    def _ratio_preference(
        reliability: torch.Tensor,
        ratios: Sequence[float],
        strength: float,
    ) -> torch.Tensor:
        """Bias high-reliability modalities toward denser retention and vice versa."""
        ratio = reliability.new_tensor(ratios)
        if ratio.numel() == 1:
            return reliability.new_zeros(reliability.size(0), reliability.size(1), 1)
        ratio = (ratio - ratio.mean()) / ratio.std(unbiased=False).clamp_min(1e-6)
        return strength * (reliability.unsqueeze(-1) - 0.5) * ratio.view(1, 1, -1)

    def forward(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        shared_language: torch.Tensor,
        temporal_rgb: Optional[torch.Tensor] = None,
        temporal_tir: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rgb_pool = rgb_search.mean(dim=1)
        tir_pool = tir_search.mean(dim=1)
        lang_pool = shared_language.mean(dim=1)

        if temporal_rgb is None or temporal_tir is None:
            temporal_pool = torch.zeros_like(rgb_pool)
        else:
            temporal_pool = 0.5 * (
                temporal_rgb.mean(dim=1) + temporal_tir.mean(dim=1)
            )

        context = torch.cat([rgb_pool, tir_pool, lang_pool, temporal_pool], dim=-1)
        context = self.context(context)
        reliability = F.softmax(self.reliability_head(context), dim=-1)

        spatial_logits = self.spatial_head(context).view(
            context.size(0), 2, len(self.spatial_ratios)
        )
        spatial_logits = spatial_logits + self._ratio_preference(
            reliability,
            self.spatial_ratios,
            self.reliability_bias_strength,
        )
        spatial_weights = F.softmax(spatial_logits, dim=-1)

        channel_logits = self.channel_head(context).view(
            context.size(0), 2, len(self.channel_ratios)
        )
        channel_logits = channel_logits + self._ratio_preference(
            reliability,
            self.channel_ratios,
            self.reliability_bias_strength,
        )
        channel_weights = F.softmax(channel_logits, dim=-1)

        spatial_ratio_tensor = reliability.new_tensor(self.spatial_ratios)
        channel_ratio_tensor = reliability.new_tensor(self.channel_ratios)
        effective_spatial = (spatial_weights * spatial_ratio_tensor).sum(dim=-1)
        effective_channel = (channel_weights * channel_ratio_tensor).sum(dim=-1)

        return {
            "reliability": reliability,
            "spatial_weights": spatial_weights,
            "channel_weights": channel_weights,
            "effective_spatial": effective_spatial,
            "effective_channel": effective_channel,
        }


class CrossModalPromptBuilder(nn.Module):
    """Visual-query / language-key-value multi-head cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.visual_norm = nn.LayerNorm(dim)
        self.language_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, visual: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        query = self.visual_norm(visual)
        key_value = self.language_norm(language)
        fused, _ = self.attn(query, key_value, key_value, need_weights=False)
        return self.out_norm(visual + fused)


class DynamicChannelSparseAttention(nn.Module):
    """Channel-transposed multi-head sparse attention inspired by AVLTrack DMSA.

    Attention is computed across channels inside every head ([d_h, d_h]) rather than
    over the spatial token dimension ([N, N]). Multiple Top-K channel ratios are
    evaluated and dynamically mixed with reliability-conditioned weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        channel_ratios: Sequence[float] = (0.5, 2.0 / 3.0, 0.75, 0.8),
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.channel_ratios = tuple(float(v) for v in channel_ratios)

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mixture_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x: [B, N, C]
            mixture_weights: [B, R], R=len(channel_ratios)
        """
        b, n, c = x.shape
        qkv = self.qkv(self.norm(x)).view(
            b, n, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        # [B, H, d_h, N], so QK^T is channel x channel.
        q = q.permute(0, 2, 3, 1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.permute(0, 2, 3, 1).contiguous()
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

        branches = []
        for ratio in self.channel_ratios:
            topk = max(1, min(self.head_dim, int(math.ceil(ratio * self.head_dim))))
            topk_idx = attn.topk(k=topk, dim=-1, largest=True, sorted=False).indices
            sparse_logits = torch.full_like(attn, torch.finfo(attn.dtype).min)
            sparse_logits.scatter_(-1, topk_idx, attn.gather(-1, topk_idx))
            sparse_attn = sparse_logits.softmax(dim=-1)
            branches.append(torch.matmul(sparse_attn, v))

        # [B, R, H, d_h, N] -> reliability-conditioned weighted mixture.
        stacked = torch.stack(branches, dim=1)
        weights = mixture_weights[:, :, None, None, None]
        out = (stacked * weights).sum(dim=1)
        out = out.permute(0, 3, 1, 2).reshape(b, n, c).contiguous()
        out = self.proj_drop(self.proj(out))
        return x + out


class ReliabilityAwareDSL(nn.Module):
    """Reliability-aware tri-modal sparse prompt generator for one interaction stage."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        spatial_ratios: Sequence[float] = (0.65, 0.75, 0.85, 1.0),
        channel_ratios: Sequence[float] = (0.5, 2.0 / 3.0, 0.75, 0.8),
        hidden_dim: int = 128,
        dropout: float = 0.0,
        reliability_bias_strength: float = 2.0,
    ):
        super().__init__()
        self.spatial_ratios = tuple(float(v) for v in spatial_ratios)
        self.channel_ratios = tuple(float(v) for v in channel_ratios)

        self.reliability = ReliabilityEstimator(
            dim=dim,
            spatial_ratios=self.spatial_ratios,
            channel_ratios=self.channel_ratios,
            hidden_dim=hidden_dim,
            reliability_bias_strength=reliability_bias_strength,
        )
        self.rgb_cross = CrossModalPromptBuilder(dim, num_heads=num_heads, dropout=dropout)
        self.tir_cross = CrossModalPromptBuilder(dim, num_heads=num_heads, dropout=dropout)
        self.rgb_dmsa = DynamicChannelSparseAttention(
            dim, num_heads=num_heads, channel_ratios=self.channel_ratios, proj_drop=dropout
        )
        self.tir_dmsa = DynamicChannelSparseAttention(
            dim, num_heads=num_heads, channel_ratios=self.channel_ratios, proj_drop=dropout
        )

    @staticmethod
    def _search_importance(
        attn: torch.Tensor,
        len_text: int,
        lens_t: int,
        lens_s: int,
        template_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Reuse MLSTTrack's query/template/text/search attention evidence."""
        lens_q = attn.shape[-1] - len_text - lens_t - lens_s
        bs, hn, _, _ = attn.shape
        search_start = lens_q + len_text + lens_t

        terms = []
        if lens_q > 0:
            terms.append(attn[:, :, :lens_q, search_start:].mean(dim=2).mean(dim=1))

        attn_t = attn[:, :, lens_q + len_text:search_start, search_start:]
        if attn_t.size(2) > 0:
            if template_mask is not None:
                mask = template_mask.to(device=attn.device, dtype=torch.bool)
                # Multi-template masks may describe only one template. Repeat when needed.
                if mask.size(1) != attn_t.size(2):
                    repeat = math.ceil(attn_t.size(2) / max(mask.size(1), 1))
                    mask = mask.repeat(1, repeat)[:, :attn_t.size(2)]
                mask = mask.unsqueeze(1).unsqueeze(-1).expand(
                    -1, hn, -1, lens_s
                )
                masked = attn_t * mask.to(attn_t.dtype)
                denom = mask.to(attn_t.dtype).sum(dim=2).clamp_min(1.0)
                terms.append((masked.sum(dim=2) / denom).mean(dim=1))
            else:
                terms.append(attn_t.mean(dim=2).mean(dim=1))

        if len_text > 0:
            terms.append(
                attn[:, :, lens_q:lens_q + len_text, search_start:]
                .mean(dim=2)
                .mean(dim=1)
            )

        terms.append(attn[:, :, search_start:, search_start:].mean(dim=2).mean(dim=1))
        importance = torch.stack(terms, dim=0).sum(dim=0)
        return importance

    def _dynamic_spatial_mask(
        self,
        attn: torch.Tensor,
        weights: torch.Tensor,
        len_text: int,
        lens_t: int,
        lens_s: int,
        template_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        importance = self._search_importance(
            attn, len_text=len_text, lens_t=lens_t, lens_s=lens_s, template_mask=template_mask
        )
        b = importance.size(0)
        branch_masks = []
        for ratio in self.spatial_ratios:
            keep = max(1, min(lens_s, int(math.ceil(ratio * lens_s))))
            idx = importance.topk(k=keep, dim=1, largest=True, sorted=False).indices
            mask = importance.new_zeros((b, lens_s))
            mask.scatter_(1, idx, 1.0)
            branch_masks.append(mask)
        branch_masks = torch.stack(branch_masks, dim=1)  # [B, R, Ns]
        soft_mask = (branch_masks * weights.unsqueeze(-1)).sum(dim=1).unsqueeze(-1)

        # Visualization-only hard removed indices. A single stage-level ratio keeps a
        # rectangular tensor while the actual forward path remains per-sample dynamic.
        stage_ratio_idx = int(weights.detach().mean(dim=0).argmax().item())
        stage_keep = max(
            1,
            min(lens_s, int(math.ceil(self.spatial_ratios[stage_ratio_idx] * lens_s))),
        )
        sorted_idx = importance.argsort(dim=1, descending=True)
        removed = sorted_idx[:, stage_keep:]
        return soft_mask, removed

    @staticmethod
    def _replace_search(
        full_tokens: torch.Tensor,
        search_tokens: torch.Tensor,
        lens_q: int,
        len_text: int,
        lens_t: int,
    ) -> torch.Tensor:
        search_start = lens_q + len_text + lens_t
        return torch.cat(
            [full_tokens[:, :search_start, :], search_tokens], dim=1
        )

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_tir: torch.Tensor,
        attn_rgb: torch.Tensor,
        attn_tir: torch.Tensor,
        lang_rgb: torch.Tensor,
        lang_tir: torch.Tensor,
        lang_shared: torch.Tensor,
        len_text: int,
        lens_t: int,
        lens_s: int,
        template_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        lens_q = x_rgb.shape[1] - len_text - lens_t - lens_s
        search_start = lens_q + len_text + lens_t
        rgb_search = x_rgb[:, search_start:, :]
        tir_search = x_tir[:, search_start:, :]
        temporal_rgb = x_rgb[:, :lens_q, :] if lens_q > 0 else None
        temporal_tir = x_tir[:, :lens_q, :] if lens_q > 0 else None

        rel = self.reliability(
            rgb_search,
            tir_search,
            lang_shared,
            temporal_rgb=temporal_rgb,
            temporal_tir=temporal_tir,
        )
        rgb_spatial_w = rel["spatial_weights"][:, 0, :]
        tir_spatial_w = rel["spatial_weights"][:, 1, :]
        rgb_channel_w = rel["channel_weights"][:, 0, :]
        tir_channel_w = rel["channel_weights"][:, 1, :]

        rgb_mask, removed_rgb = self._dynamic_spatial_mask(
            attn_rgb, rgb_spatial_w, len_text, lens_t, lens_s, template_mask
        )
        tir_mask, removed_tir = self._dynamic_spatial_mask(
            attn_tir, tir_spatial_w, len_text, lens_t, lens_s, template_mask
        )

        # MHCA first constructs vision-language cues; spatial and channel sparsity then
        # suppress currently mismatched content and produce sparse semantic prompts.
        rgb_cue = self.rgb_cross(rgb_search, lang_rgb) * rgb_mask
        tir_cue = self.tir_cross(tir_search, lang_tir) * tir_mask
        rgb_prompt = self.rgb_dmsa(rgb_cue, rgb_channel_w)
        tir_prompt = self.tir_dmsa(tir_cue, tir_channel_w)

        x_rgb_sparse = self._replace_search(
            x_rgb, rgb_prompt, lens_q=lens_q, len_text=len_text, lens_t=lens_t
        )
        x_tir_sparse = self._replace_search(
            x_tir, tir_prompt, lens_q=lens_q, len_text=len_text, lens_t=lens_t
        )

        stats = {
            "reliability_rgb": rel["reliability"][:, 0],
            "reliability_tir": rel["reliability"][:, 1],
            "effective_spatial_rgb": rel["effective_spatial"][:, 0],
            "effective_spatial_tir": rel["effective_spatial"][:, 1],
            "effective_channel_rgb": rel["effective_channel"][:, 0],
            "effective_channel_tir": rel["effective_channel"][:, 1],
            "spatial_weights_rgb": rgb_spatial_w,
            "spatial_weights_tir": tir_spatial_w,
            "channel_weights_rgb": rgb_channel_w,
            "channel_weights_tir": tir_channel_w,
        }
        return (
            x_rgb_sparse,
            x_tir_sparse,
            rgb_prompt,
            tir_prompt,
            removed_rgb,
            removed_tir,
            stats,
        )


class MultiLevelPromptModulator(nn.Module):
    """Zero-initialized residual prompt reweighting for a search-token stream."""

    def __init__(self, dim: int, init_scale: float = 0.0):
        super().__init__()
        self.prompt_norm = nn.LayerNorm(dim)
        self.selector = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        full_tokens: torch.Tensor,
        prompt: torch.Tensor,
        reliability: Optional[torch.Tensor],
        len_text: int,
        lens_t: int,
        lens_s: int,
    ) -> torch.Tensor:
        lens_q = full_tokens.shape[1] - len_text - lens_t - lens_s
        search_start = lens_q + len_text + lens_t
        search = full_tokens[:, search_start:, :]
        selector = torch.tanh(self.selector(self.prompt_norm(prompt)))
        if reliability is not None:
            selector = selector * reliability.view(-1, 1, 1)
        enhanced = search + self.scale * selector * search
        return torch.cat([full_tokens[:, :search_start, :], enhanced], dim=1)
