import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import Mlp
from lib.models.layers.attn import Attention_qkv
from functools import partial
import torch.nn.functional as F
from lib.models.mlsttrack.itpn import fast_itpn_base_3324_patch16_224
from lib.models.mlsttrack.dual_memory import DualTimescaleMemoryBank, MemoryConfig

class ATTFu(nn.Module):
    def __init__(self, channels, ratio=1.0):
        super(ATTFu, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(4 * channels, int(ratio * channels), bias=False),
            nn.ReLU(),
            nn.Linear(int(ratio * channels), 4 * channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, l_pro, l_text, l_tem, l_tem_d):
        l_tem = torch.mean(l_tem, dim=1, keepdim=True)
        l_tem_d = torch.mean(l_tem_d, dim=1, keepdim=True)
        l_fu = torch.cat([l_pro, l_text, l_tem, l_tem_d], dim=2)
        att = self.fc(l_fu)

        return att[:, :, :self.channels]

class MLSTTrack(nn.Module):

    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER"):

        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.box_head = box_head
        self.offset_range_factor = cfg.MODEL.TSG.LAYER
        self.track_query_len = cfg.MODEL.TSG.TRACK_QUERY
        self.track_beforequery_len = cfg.MODEL.TSG.TRACK_QUERY_OLD
        self.template_number = cfg.DATA.TEMPLATE.NUMBER
        self.decode_fuse_search = conv(hidden_dim * 2, hidden_dim)

        self.prompt = ATTFu(hidden_dim)

        # Uncertainty-aware dual-timescale semantic memory.
        mem_cfg = cfg.MODEL.MEMORY
        memory_config = MemoryConfig(
            long_capacity=mem_cfg.LONG_CAPACITY,
            short_capacity=mem_cfg.SHORT_CAPACITY,
            long_topk=mem_cfg.LONG_TOPK,
            short_topk=mem_cfg.SHORT_TOPK,
            long_novelty=mem_cfg.LONG_NOVELTY,
            long_quality_threshold=mem_cfg.LONG_QUALITY_THRESHOLD,
            long_uncertainty_threshold=mem_cfg.LONG_UNCERTAINTY_THRESHOLD,
            short_quality_threshold=mem_cfg.SHORT_QUALITY_THRESHOLD,
            quality_weight=mem_cfg.QUALITY_WEIGHT,
            uncertainty_weight=mem_cfg.UNCERTAINTY_WEIGHT,
            recency_weight=mem_cfg.RECENCY_WEIGHT,
            recency_tau=mem_cfg.RECENCY_TAU,
        )
        self.memory_enabled = mem_cfg.ENABLED
        self.memory_r = DualTimescaleMemoryBank(hidden_dim, memory_config)
        self.memory_x = DualTimescaleMemoryBank(hidden_dim, memory_config)
        self.memory_step = 0

        self.text_strengthen_r = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.text_strengthen_x = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)

        # A learned localization-uncertainty head. Its prediction is calibrated
        # against 1-IoU during training and blended with a score-map heuristic
        # for robust early-stage memory gating.
        uncertainty_hidden = max(hidden_dim // 4, 32)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, uncertainty_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(uncertainty_hidden, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_heuristic_weight = mem_cfg.HEURISTIC_WEIGHT

        self.CSS_strengthen_r = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_r = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)
        self.CSS_strengthen_x = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process_x = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                                 drop=0.)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text_label = None,
                ce_template_mask = None,
                return_last_attn = False,
                track_query_before = None,
                ):

        out_dict = []
        if track_query_before is None:
            self.reset_memory()
        for i in range(len(search)):
            x, aux_dict, len_zx = self.backbone(z=template, x=search[i], track_query_before=track_query_before,
                                                text_label=text_label[i],
                                                ce_template_mask=ce_template_mask,
                                                return_last_attn=return_last_attn, )
            num_template_token = len_zx[0]
            num_search_token = len_zx[1]
            B, N, C = x.size()
            temp_r = x[:, :N // 2, :]
            temp_x = x[:, N // 2:, :]
            text_r = temp_r[:, self.track_query_len:self.track_query_len + 1, :]
            text_x = temp_x[:, self.track_query_len:self.track_query_len + 1, :]
            temp_r_str = self.prompt(temp_r[:, :self.track_query_len, :],
                                     text_r,
                                     temp_r[:, self.track_query_len + 1:num_template_token // 2 + self.track_query_len + 1, :],
                                     temp_r[:, num_template_token // 2 + self.track_query_len + 1:num_template_token +self.track_query_len + 1,:])
            temp_x_str = self.prompt(temp_x[:, :self.track_query_len, :],
                                     text_x,
                                     temp_x[:, self.track_query_len + 1:num_template_token // 2 + self.track_query_len + 1, :],
                                     temp_x[:, num_template_token // 2 + self.track_query_len + 1:num_template_token + self.track_query_len + 1,:])
            temp_r_query = temp_r_str.clone().detach()
            temp_x_query = temp_x_str.clone().detach()
            track_query_before = [temp_r_query, temp_x_query]

            # Build a joint visual-language query before memory augmentation.
            feat_last_r = temp_r[:, -num_search_token:, :]
            feat_last_x = temp_x[:, -num_search_token:, :]
            visual_state_r = feat_last_r.mean(dim=1, keepdim=True)
            visual_state_x = feat_last_x.mean(dim=1, keepdim=True)
            memory_key_r = F.normalize(text_r + visual_state_r, dim=-1)
            memory_key_x = F.normalize(text_x + visual_state_x, dim=-1)
            identity_value_r = text_r
            identity_value_x = text_x
            state_value_r = F.layer_norm(text_r + visual_state_r, (C,))
            state_value_x = F.layer_norm(text_x + visual_state_x, (C,))

            novelty_r = torch.ones(B, 1, device=x.device, dtype=x.dtype)
            novelty_x = torch.ones(B, 1, device=x.device, dtype=x.dtype)
            if self.memory_enabled:
                retrieved_r, retrieved_mask_r, novelty_r = self.memory_r.retrieve(
                    memory_key_r, timestamp=self.memory_step
                )
                if retrieved_r is not None:
                    feat_last_r = feat_last_r + self.text_strengthen_r(
                        feat_last_r, retrieved_r, retrieved_r, mask=retrieved_mask_r
                    )

                retrieved_x, retrieved_mask_x, novelty_x = self.memory_x.retrieve(
                    memory_key_x, timestamp=self.memory_step
                )
                if retrieved_x is not None:
                    feat_last_x = feat_last_x + self.text_strengthen_x(
                        feat_last_x, retrieved_x, retrieved_x, mask=retrieved_mask_x
                    )

            temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_last_r, feat_last_r)
            temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
            att_r = torch.matmul(feat_last_r, temp_attn_r.transpose(1, 2))
            feat_last_r = att_r * feat_last_r

            temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_last_x, feat_last_x)
            temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
            att_x = torch.matmul(feat_last_x, temp_attn_x.transpose(1, 2))
            feat_last_x = att_x * feat_last_x

            feat_last = torch.cat([feat_last_r, feat_last_x], dim=-1)

            out = self.forward_head(feat_last, None)
            out['memory_novelty'] = 0.5 * (novelty_r + novelty_x)

            # Memory is updated only after the prediction head has estimated
            # localization quality and uncertainty for the current frame.
            if self.memory_enabled:
                self.memory_r.write(
                    memory_key_r, identity_value_r, state_value_r,
                    out['memory_quality'], out['memory_uncertainty'],
                    timestamp=self.memory_step,
                )
                self.memory_x.write(
                    memory_key_x, identity_value_x, state_value_x,
                    out['memory_quality'], out['memory_uncertainty'],
                    timestamp=self.memory_step,
                )
            self.memory_step += 1

            out.update(aux_dict)
            out['track_query_before'] = track_query_before
            out['backbone_feat'] = feat_last
            out_dict.append(out)

        return out_dict

    def reset_memory(self):
        self.memory_r.reset()
        self.memory_x.reset()
        self.memory_step = 0

    @staticmethod
    def _score_map_uncertainty(score_map: torch.Tensor) -> torch.Tensor:
        """Normalized spatial entropy plus peak ambiguity in [0, 1]."""
        eps = 1e-6
        flat = score_map.flatten(1).clamp(min=eps, max=1.0 - eps)
        spatial_prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(eps)
        entropy = -(spatial_prob * spatial_prob.log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(max(flat.shape[1], 2))
        if flat.shape[1] >= 2:
            top2 = flat.topk(k=2, dim=1).values
            peak_margin = (top2[:, :1] - top2[:, 1:2]) / top2[:, :1].clamp_min(eps)
            ambiguity = 1.0 - peak_margin.clamp(0.0, 1.0)
        else:
            ambiguity = 1.0 - flat
        return (0.5 * entropy + 0.5 * ambiguity).clamp(0.0, 1.0)

    def forward_head(self, cat_feature ,gt_score_map=None):

        opt = (cat_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        # HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.decode_fuse_search(opt_feat)
        # bs, C, _, _ = opt_feat.size()
        # Nq = 1

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            uncertainty_pred = self.uncertainty_head(opt_feat)
            heuristic_uncertainty = self._score_map_uncertainty(score_map_ctr)
            memory_uncertainty = (
                (1.0 - self.uncertainty_heuristic_weight) * uncertainty_pred
                + self.uncertainty_heuristic_weight * heuristic_uncertainty
            ).clamp(0.0, 1.0)
            confidence = score_map_ctr.flatten(1).amax(dim=1, keepdim=True)
            memory_quality = (confidence * (1.0 - memory_uncertainty)).clamp(0.0, 1.0)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'uncertainty_pred': uncertainty_pred,
                   'heuristic_uncertainty': heuristic_uncertainty,
                   'memory_uncertainty': memory_uncertainty,
                   'memory_quality': memory_quality}
            return out
        else:
            raise NotImplementedError


def build_mlsttrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('SOT' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, cfg=cfg)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = MLSTTrack(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'SOT' in cfg.MODEL.PRETRAIN_FILE and training:
        # checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        # print(f"pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE):{pretrained_file}") # 查看路径
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('22222222222222  Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        # print('missing_keys, unexpected_keys', missing_keys, unexpected_keys)
    return model

