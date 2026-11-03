"""Uncertainty-aware dual-timescale memory for MLSTTrack.

The bank keeps two complementary memories for every item in the batch:
1) identity memory: sparse, reliable and semantically diverse long-term anchors;
2) state memory: a FIFO queue of recent target states.

The class intentionally does not inherit from ``nn.Module`` because the memory is
online state rather than a learnable model parameter. All written tensors are
``detach``-ed to avoid retaining computation graphs across video frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryConfig:
    long_capacity: int = 4
    short_capacity: int = 8
    long_topk: int = 2
    short_topk: int = 2
    long_novelty: float = 0.92
    long_quality_threshold: float = 0.55
    long_uncertainty_threshold: float = 0.45
    short_quality_threshold: float = 0.20
    quality_weight: float = 0.35
    uncertainty_weight: float = 0.50
    recency_weight: float = 0.20
    recency_tau: float = 5.0


class DualTimescaleMemoryBank:
    """Batched long/short-term memory with reliability-aware retrieval.

    Shapes:
        key/value: [B, 1, C]
        quality/uncertainty: [B, 1]
    """

    def __init__(self, dim: int, cfg: MemoryConfig):
        self.dim = dim
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.batch_size: Optional[int] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.step = 0

        self.long_keys = None
        self.long_values = None
        self.long_quality = None
        self.long_uncertainty = None
        self.long_time = None
        self.long_valid = None

        self.short_keys = None
        self.short_values = None
        self.short_quality = None
        self.short_uncertainty = None
        self.short_time = None
        self.short_valid = None
        self.short_ptr = None

    def _ensure_initialized(self, key: torch.Tensor) -> None:
        batch_size = key.shape[0]
        if self.batch_size is not None:
            if self.batch_size != batch_size:
                raise RuntimeError(
                    f"Memory batch size changed from {self.batch_size} to {batch_size}. "
                    "Call reset() at the beginning of each sequence/batch."
                )
            return

        self.batch_size = batch_size
        self.device = key.device
        self.dtype = key.dtype
        b, c = batch_size, self.dim
        lc, sc = self.cfg.long_capacity, self.cfg.short_capacity

        self.long_keys = torch.zeros(b, lc, c, device=key.device, dtype=key.dtype)
        self.long_values = torch.zeros_like(self.long_keys)
        self.long_quality = torch.zeros(b, lc, device=key.device, dtype=key.dtype)
        self.long_uncertainty = torch.ones(b, lc, device=key.device, dtype=key.dtype)
        self.long_time = torch.zeros(b, lc, device=key.device, dtype=torch.long)
        self.long_valid = torch.zeros(b, lc, device=key.device, dtype=torch.bool)

        self.short_keys = torch.zeros(b, sc, c, device=key.device, dtype=key.dtype)
        self.short_values = torch.zeros_like(self.short_keys)
        self.short_quality = torch.zeros(b, sc, device=key.device, dtype=key.dtype)
        self.short_uncertainty = torch.ones(b, sc, device=key.device, dtype=key.dtype)
        self.short_time = torch.zeros(b, sc, device=key.device, dtype=torch.long)
        self.short_valid = torch.zeros(b, sc, device=key.device, dtype=torch.bool)
        self.short_ptr = torch.zeros(b, device=key.device, dtype=torch.long)

    @staticmethod
    def _flatten_token(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, C], got {tuple(x.shape)}")
        return x[:, 0]

    def _write_slot(
        self,
        bank: str,
        batch_index: int,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
        quality: torch.Tensor,
        uncertainty: torch.Tensor,
        timestamp: int,
    ) -> None:
        getattr(self, f"{bank}_keys")[batch_index, slot].copy_(key)
        getattr(self, f"{bank}_values")[batch_index, slot].copy_(value)
        getattr(self, f"{bank}_quality")[batch_index, slot] = quality
        getattr(self, f"{bank}_uncertainty")[batch_index, slot] = uncertainty
        getattr(self, f"{bank}_time")[batch_index, slot] = timestamp
        getattr(self, f"{bank}_valid")[batch_index, slot] = True

    @torch.no_grad()
    def write(
        self,
        key: torch.Tensor,
        identity_value: torch.Tensor,
        state_value: torch.Tensor,
        quality: torch.Tensor,
        uncertainty: torch.Tensor,
        timestamp: Optional[int] = None,
    ) -> None:
        """Write current state into long- and short-term memories.

        Slot 0 of identity memory is a permanent anchor. Later identity entries
        must be reliable and sufficiently novel. State memory uses a reliable
        FIFO policy so it can follow short-term appearance changes.
        """
        key = F.normalize(self._flatten_token(key.detach()), dim=-1)
        identity_value = self._flatten_token(identity_value.detach())
        state_value = self._flatten_token(state_value.detach())
        quality = quality.detach().view(-1).clamp(0.0, 1.0)
        uncertainty = uncertainty.detach().view(-1).clamp(0.0, 1.0)
        self._ensure_initialized(key.unsqueeze(1))

        if timestamp is None:
            timestamp = self.step
        self.step = max(self.step, int(timestamp) + 1)

        for b in range(self.batch_size):
            # The first observation is the immutable identity anchor.
            if not self.long_valid[b].any():
                self._write_slot(
                    "long", b, 0, key[b], identity_value[b],
                    torch.ones_like(quality[b]), torch.zeros_like(uncertainty[b]), timestamp,
                )
            else:
                valid_idx = self.long_valid[b].nonzero(as_tuple=False).flatten()
                max_sim = F.cosine_similarity(
                    key[b].unsqueeze(0), self.long_keys[b, valid_idx], dim=-1
                ).max()
                reliable = (
                    quality[b] >= self.cfg.long_quality_threshold
                    and uncertainty[b] <= self.cfg.long_uncertainty_threshold
                )
                novel = max_sim < self.cfg.long_novelty
                if reliable and novel:
                    free = (~self.long_valid[b]).nonzero(as_tuple=False).flatten()
                    if free.numel() > 0:
                        slot = int(free[0])
                    else:
                        # Keep slot 0 forever. Replace the least useful remaining anchor.
                        candidates = torch.arange(1, self.cfg.long_capacity, device=key.device)
                        utility = (
                            self.long_quality[b, candidates]
                            - self.long_uncertainty[b, candidates]
                        )
                        slot = int(candidates[utility.argmin()])
                        new_utility = quality[b] - uncertainty[b]
                        if new_utility <= utility.min():
                            slot = -1
                    if slot >= 0:
                        self._write_slot(
                            "long", b, slot, key[b], identity_value[b],
                            quality[b], uncertainty[b], timestamp,
                        )

            # Recent-state memory is updated more frequently, but still rejects
            # very unreliable observations.
            if quality[b] >= self.cfg.short_quality_threshold:
                slot = int(self.short_ptr[b])
                self._write_slot(
                    "short", b, slot, key[b], state_value[b],
                    quality[b], uncertainty[b], timestamp,
                )
                self.short_ptr[b] = (slot + 1) % self.cfg.short_capacity

    def _bank_scores(
        self,
        bank: str,
        query: torch.Tensor,
        batch_index: int,
        timestamp: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = getattr(self, f"{bank}_valid")[batch_index]
        valid_idx = valid.nonzero(as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return valid_idx, torch.empty(0, device=query.device, dtype=query.dtype)

        keys = getattr(self, f"{bank}_keys")[batch_index, valid_idx]
        quality = getattr(self, f"{bank}_quality")[batch_index, valid_idx]
        uncertainty = getattr(self, f"{bank}_uncertainty")[batch_index, valid_idx]
        similarity = F.cosine_similarity(query[batch_index].unsqueeze(0), keys, dim=-1)
        scores = (
            similarity
            + self.cfg.quality_weight * quality
            - self.cfg.uncertainty_weight * uncertainty
        )
        if bank == "short":
            memory_time = getattr(self, f"{bank}_time")[batch_index, valid_idx]
            age = (timestamp - memory_time).clamp(min=0).to(query.dtype)
            recency = torch.exp(-age / max(self.cfg.recency_tau, 1e-6))
            scores = scores + self.cfg.recency_weight * recency
        return valid_idx, scores

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        timestamp: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Retrieve reliable long/short memories.

        Returns:
            tokens: [B, K, C] or None
            padding_mask: [B, K], True means padding/ignored
            novelty: [B, 1], one minus maximum key similarity
        """
        query = F.normalize(self._flatten_token(query.detach()), dim=-1)
        if self.batch_size is None:
            novelty = torch.ones(query.shape[0], 1, device=query.device, dtype=query.dtype)
            return None, None, novelty
        if timestamp is None:
            timestamp = self.step

        total_k = self.cfg.long_topk + self.cfg.short_topk
        tokens = torch.zeros(
            self.batch_size, total_k, self.dim, device=query.device, dtype=query.dtype
        )
        padding_mask = torch.ones(
            self.batch_size, total_k, device=query.device, dtype=torch.bool
        )
        novelty = torch.ones(self.batch_size, 1, device=query.device, dtype=query.dtype)

        for b in range(self.batch_size):
            cursor = 0
            all_similarities = []
            for bank, topk in (("long", self.cfg.long_topk), ("short", self.cfg.short_topk)):
                valid_idx, scores = self._bank_scores(bank, query, b, int(timestamp))
                if valid_idx.numel() == 0 or topk <= 0:
                    continue
                k = min(topk, valid_idx.numel())
                selected_local = scores.topk(k=k, largest=True).indices
                selected_slots = valid_idx[selected_local]
                values = getattr(self, f"{bank}_values")[b, selected_slots]
                tokens[b, cursor:cursor + k] = values
                padding_mask[b, cursor:cursor + k] = False
                cursor += k

                keys = getattr(self, f"{bank}_keys")[b, valid_idx]
                all_similarities.append(
                    F.cosine_similarity(query[b].unsqueeze(0), keys, dim=-1)
                )

            if all_similarities:
                novelty[b, 0] = 0.5 * (1.0 - torch.cat(all_similarities).max().clamp(-1.0, 1.0))

        # This can happen only before the first write.
        if padding_mask.all():
            return None, None, novelty
        return tokens, padding_mask, novelty
