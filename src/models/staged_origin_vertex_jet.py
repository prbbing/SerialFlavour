"""
Staged three-encoder transformer jet-flavour classifier.

Architecture
------------
Stage 1 (encoder 1) — track-origin prediction.
    All 21 track features → input_proj1 → Encoder1 → origin_head → soft_probs.
    Per-track task; no CLS token.

Stage 2 (encoder 2) — differentiable origin-gated secondary-vertex fit.
    soft_probs → class_embed → soft_embed (weighted sum of learnable class vectors).
    soft_embed + vertex_features → input_proj2 → Encoder2 → vertex_weight_head.
    For each vertex leg k:
        w_i = p_origin_k(track_i) · refine_k(track_i) / σ_d0,i²
        L̂xy_k = (Σ w_i |d0_i|) / (Σ w_i)        closed-form, differentiable.
        dẑ_k  = (Σ w_i · z0sinθ_i / σ²) / (Σ w_i / σ²)     signed weighted mean.
    Calibration: pred *= exp(log_scale) per leg per coordinate.

Stage 3 (encoder 3) — jet-flavour classification.
    Fitted Lxy/dz → vertex_embed → vertex tokens.
    vertex tokens + CLS token + (filtered) track features → Encoder3.
    CLS output → jet_head → jet_logits (b / c / light).

The three stages communicate only through differentiable intermediates
(soft_probs, fitted vertex coordinates), enabling end-to-end training with
a combined loss:
    λ_jet·CE(jet) + λ_origin·CE(origin) + λ_vertex·Σ Lxy_loss
"""
import torch
import torch.nn as nn


class StagedOriginVertexJetTransformer(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_origin_classes, n_jet_classes,
                 vertex_feat_indices, d0_idx, d0_unc_idx, dphi_idx,
                 z0st_idx, z0st_unc_idx,
                 vertex_leg_origin_matrix, gate_temp=0.1,
                 fit_lxy=True, fit_dz=True,
                 stage3_use_origin_probs=False, stage3_use_vtx_weight=False,
                 tagging_feat_indices=None,
                 calibrate_vertex_fit=True):
        super().__init__()
        self.n_origin_classes = n_origin_classes     # 8
        self.n_vertex_legs    = vertex_leg_origin_matrix.shape[1]  # 2 or 3
        self.gate_temp        = gate_temp
        self.fit_lxy          = fit_lxy
        self.fit_dz           = fit_dz
        self.stage3_use_origin_probs = stage3_use_origin_probs
        self.stage3_use_vtx_weight   = stage3_use_vtx_weight
        self.calibrate_vertex_fit    = calibrate_vertex_fit
        n_vtx_coords          = int(fit_lxy) + int(fit_dz)  # 1 or 2

        # -- learnable per-leg multiplicative calibration -----------------
        # Initialised to exp(0) = 1.0 (no-op at init).
        if calibrate_vertex_fit and fit_lxy:
            self.lxy_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))
        if calibrate_vertex_fit and fit_dz:
            self.dz_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))

        # ================================================================
        # Stage 1 — track-origin prediction
        # ================================================================
        self.input_proj1 = nn.Linear(in_dim, d_model)
        _enc1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder1 = nn.TransformerEncoder(_enc1_layer, num_layers=n_layers)
        self.origin_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_origin_classes))

        # Learnable per-class embedding vectors.
        # soft_embed = soft_probs @ class_embed → (B, K, d_model).
        self.class_embed = nn.Parameter(torch.zeros(n_origin_classes, d_model))
        nn.init.trunc_normal_(self.class_embed, std=0.02)

        # ================================================================
        # Stage 2 — differentiable secondary-vertex fit
        # ================================================================
        n_vtx_feats = len(vertex_feat_indices)
        self.input_proj2 = nn.Linear(n_vtx_feats, d_model)
        _enc2_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder2 = nn.TransformerEncoder(_enc2_layer, num_layers=n_layers)
        # per-leg refinement weight ∈ [0, 1]
        self.vertex_weight_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, self.n_vertex_legs))

        # ================================================================
        # Stage 3 — jet-flavour classification
        # ================================================================
        if tagging_feat_indices is None:
            tagging_feat_indices = list(range(in_dim))
        self.register_buffer("tagging_feat_idx",
            torch.tensor(tagging_feat_indices, dtype=torch.long))
        # input dim may optionally include origin probs + vtx weight
        _stage3_in_dim = len(tagging_feat_indices)
        if stage3_use_origin_probs:
            _stage3_in_dim += n_origin_classes
        if stage3_use_vtx_weight:
            _stage3_in_dim += self.n_vertex_legs
        self.input_proj3 = nn.Linear(_stage3_in_dim, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # vertex summary: raw coords → d_model token
        self.vertex_embed = nn.Sequential(
            nn.Linear(n_vtx_coords, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

        _enc3_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder3 = nn.TransformerEncoder(_enc3_layer, num_layers=n_layers)
        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

        # non-trainable indices
        self.register_buffer("vertex_feat_idx",
            torch.tensor(vertex_feat_indices, dtype=torch.long))
        self.register_buffer("vertex_leg_origin_matrix",
            torch.tensor(vertex_leg_origin_matrix, dtype=torch.float32))
        self.d0_idx, self.d0_unc_idx, self.dphi_idx = d0_idx, d0_unc_idx, dphi_idx
        self.z0st_idx, self.z0st_unc_idx = z0st_idx, z0st_unc_idx

    def forward(self, x, mask):
        """Full three-stage forward pass.

        Args:
            x:    (B, K, F)      — track features
            mask: (B, K)  bool   — track validity

        Returns:
            dict: jet_logits (B,3), origin_logits (B,K,8),
                  vtx_weight (B,K,L), lxy_pred (B,L), dz_pred (B,L)
        """
        B, K, _ = x.shape
        track_padding_mask = ~mask
        mask_f = mask.unsqueeze(-1).float()

        # ---- Stage 1: track-origin prediction ----
        h1 = self.input_proj1(x)                               # (B, K, D)
        h1 = self.encoder1(h1, src_key_padding_mask=track_padding_mask)
        origin_logits = self.origin_head(h1)                   # (B, K, 8)
        soft_probs = torch.softmax(origin_logits, dim=-1) * mask_f  # zero padding

        # ---- Stage 2: differentiable vertex fit ----
        # soft origin embedding via learnable class vectors
        soft_embed = soft_probs @ self.class_embed             # (B, K, D)
        vtx_feats  = x.index_select(-1, self.vertex_feat_idx)  # vertex-specific features
        h2 = self.input_proj2(vtx_feats) + soft_embed
        h2 = self.encoder2(h2, src_key_padding_mask=track_padding_mask)

        # per-leg gating: refine (learnt) × origin gate (from Stage 1 soft probs)
        refine = torch.sigmoid(self.vertex_weight_head(h2))    # (B, K, L)
        leg_origin_probs = soft_probs @ self.vertex_leg_origin_matrix  # (B, K, L)
        gate = torch.sigmoid((leg_origin_probs - 0.5) / self.gate_temp)
        vtx_weight = refine * gate * mask_f                    # (B, K, L)

        # -- closed-form Lxy fit (weighted least-squares) --
        # Principle: |d0_i| ≈ Lxy · |sin Δφ_i|
        #   1) estimate flight φ direction from circular mean of dphi
        #   2) closed-form weighted mean: Lxy = Σ(w_i·|sinΔφ_i|·|d0_i|/σ²) / Σ(w_i·sin²Δφ_i/σ²)
        if self.fit_lxy:
            d0      = x[..., self.d0_idx].unsqueeze(-1)
            d0_unc  = x[..., self.d0_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            dphi    = x[..., self.dphi_idx].unsqueeze(-1)
            sin_dphi   = torch.sin(dphi)
            cos_dphi   = torch.cos(dphi)
            sum_sin    = (vtx_weight * sin_dphi).sum(1)       # (B, L)
            sum_cos    = (vtx_weight * cos_dphi).sum(1)
            flight_phi = torch.atan2(sum_sin, sum_cos).unsqueeze(1)  # (B, 1, L)
            delta_phi  = dphi - flight_phi
            sin_d      = torch.sin(delta_phi)
            inv_var_d0 = vtx_weight / d0_unc.pow(2)           # 1/σ² weights
            num_lxy    = (inv_var_d0 * sin_d.abs() * d0.abs()).sum(1)
            den_lxy    = (inv_var_d0 * sin_d.pow(2)).sum(1).clamp(min=1e-6)
            lxy_pred   = num_lxy / den_lxy                    # (B, L)
            if self.calibrate_vertex_fit:
                lxy_pred = lxy_pred * torch.exp(self.lxy_log_scale).unsqueeze(0)
        else:
            lxy_pred = vtx_weight.new_zeros(vtx_weight.shape[0], self.n_vertex_legs)

        # -- closed-form dz fit (signed inverse-variance weighted mean) --
        # Principle: z0·sinθ ≈ dz
        if self.fit_dz:
            z0st     = x[..., self.z0st_idx].unsqueeze(-1)
            z0st_unc = x[..., self.z0st_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            inv_var_z0 = vtx_weight / z0st_unc.pow(2)
            num_dz     = (inv_var_z0 * z0st).sum(1)
            den_dz     = inv_var_z0.sum(1).clamp(min=1e-6)
            dz_pred    = num_dz / den_dz  # signed
            if self.calibrate_vertex_fit:
                dz_pred = dz_pred * torch.exp(self.dz_log_scale).unsqueeze(0)
        else:
            dz_pred = vtx_weight.new_zeros(vtx_weight.shape[0], self.n_vertex_legs)

        # ---- Stage 3: jet-flavour classification ----
        # build vertex summary tokens (log1p-compressed coordinates → d_model)
        _vtx_parts = []
        if self.fit_lxy:
            _vtx_parts.append(torch.log1p(lxy_pred.clamp(min=0)))
        if self.fit_dz:
            _vtx_parts.append(torch.log1p(dz_pred.abs()))
        vtx_summary = torch.stack(_vtx_parts, dim=-1)         # (B, L, N_VTX_COORDS)
        vtx_tokens  = self.vertex_embed(vtx_summary)          # (B, L, D)

        # select tagging_fields subset, optionally append Stage 1/2 signals
        x_tag = x.index_select(-1, self.tagging_feat_idx)
        _h3_input_parts = [x_tag]
        if self.stage3_use_origin_probs:
            _h3_input_parts.append(soft_probs)
        if self.stage3_use_vtx_weight:
            _h3_input_parts.append(vtx_weight)
        h3_input  = torch.cat(_h3_input_parts, dim=-1) if len(_h3_input_parts) > 1 else x_tag
        h3_tracks = self.input_proj3(h3_input)

        # prepend CLS + vertex tokens to track sequence
        cls_tok = self.cls_token.expand(B, -1, -1)
        h3_in   = torch.cat([cls_tok, vtx_tokens, h3_tracks], dim=1)  # (B, 1+L+K, D)

        extra_valid = torch.ones(B, 1 + self.n_vertex_legs, dtype=torch.bool,
                                 device=x.device)
        src_key_padding_mask3 = ~torch.cat([extra_valid, mask], dim=1)

        h3 = self.encoder3(h3_in, src_key_padding_mask=src_key_padding_mask3)
        jet_logits = self.jet_head(h3[:, 0])     # CLS position only

        return {
            "jet_logits":        jet_logits,          # (B, 3)
            "origin_logits":     origin_logits,       # (B, K, 8)
            "vtx_weight":        vtx_weight,          # (B, K, L)
            "leg_origin_probs":  leg_origin_probs,    # (B, K, L)
            "gate":              gate,                # (B, K, L)
            "refine":            refine,              # (B, K, L)
            "lxy_pred":          lxy_pred,            # (B, L)
            "dz_pred":           dz_pred,             # (B, L)
        }

    def calibration_scales(self):
        """Return current learned exp(log_scale) values per leg, for logging."""
        out = {}
        if self.calibrate_vertex_fit and self.fit_lxy:
            out["lxy"] = torch.exp(self.lxy_log_scale).detach().cpu().tolist()
        if self.calibrate_vertex_fit and self.fit_dz:
            out["dz"] = torch.exp(self.dz_log_scale).detach().cpu().tolist()
        return out


def build_staged_origin_vertex_jet(config):
    """Factory: Config → StagedOriginVertexJetTransformer."""
    return StagedOriginVertexJetTransformer(
        in_dim=config.n_feats, d_model=config.d_model,
        n_heads=config.n_heads, n_layers=config.n_layers,
        d_ffn=config.d_ffn, dropout=config.dropout,
        n_origin_classes=config.n_origin_classes,
        n_jet_classes=config.n_jet_classes,
        vertex_feat_indices=config.vertex_feat_idx,
        d0_idx=config.d0_idx, d0_unc_idx=config.d0_unc_idx,
        dphi_idx=config.dphi_idx,
        z0st_idx=config.z0st_idx, z0st_unc_idx=config.z0st_unc_idx,
        vertex_leg_origin_matrix=config.vertex_leg_origin_matrix,
        gate_temp=config.gate_temp,
        fit_lxy=config.fit_lxy, fit_dz=config.fit_dz,
        stage3_use_origin_probs=config.stage3_use_origin_probs,
        stage3_use_vtx_weight=config.stage3_use_vtx_weight,
        tagging_feat_indices=config.tagging_feat_idx,
        calibrate_vertex_fit=config.calibrate_vertex_fit,
    )
