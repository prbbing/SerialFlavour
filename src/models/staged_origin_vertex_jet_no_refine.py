"""
Staged two-encoder transformer jet-flavour classifier (no refine).

Differs from staged_origin_vertex_jet in two ways:
  1. Encoder2 (refine head) is removed — vtx_weight = gate * mask_f only.
  2. Vertex fitting uses selectable formulas (old_dz / two_step / wls_3d).

Three vertex-fitting methods, selected via `vertex_fit_method`:
  "old_dz"    — original staged formula: Lxy closed-form fit, dz = Σ(w·z0st/σ²)/Σ(w/σ²)
  "two_step"  — keep existing Lxy/flight_phi fit, replace dz with
                two-step WLS using the geometric relation
  "wls_3d"    — joint 3D WLS solving (X,Y,Z) from all tracks
"""
import torch
import torch.nn as nn

MIN_SIN_THETA = 0.1


class StagedOriginVertexJetTransformerNoRefine(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_origin_classes, n_jet_classes,
                 vertex_feat_indices, d0_idx, d0_unc_idx, dphi_idx,
                 z0st_idx, z0st_unc_idx, theta_idx,
                 vertex_leg_origin_matrix, gate_temp=0.1,
                 vertex_fit_method="wls_3d", vertex_fit_reg=1e-6,
                 fit_lxy=True, fit_dz=True,
                 stage3_use_origin_probs=False, stage3_use_vtx_weight=False,
                 stage3_use_vertex_tokens=True,
                 tagging_feat_indices=None,
                 calibrate_vertex_fit=True):
        super().__init__()
        self.n_origin_classes = n_origin_classes
        self.n_vertex_legs    = vertex_leg_origin_matrix.shape[1]
        self.gate_temp        = gate_temp
        self.fit_lxy          = fit_lxy
        self.fit_dz           = fit_dz
        self.stage3_use_origin_probs = stage3_use_origin_probs
        self.stage3_use_vtx_weight   = stage3_use_vtx_weight
        self.stage3_use_vertex_tokens = stage3_use_vertex_tokens
        self.calibrate_vertex_fit    = calibrate_vertex_fit
        self.vertex_fit_method = vertex_fit_method
        self.vertex_fit_reg    = vertex_fit_reg
        n_vtx_coords          = int(fit_lxy) + int(fit_dz)

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

        # ================================================================
        # Stage 2 — no refine encoder; gate-only vertex weighting
        # ================================================================
        # No encoder2, class_embed, or vertex_weight_head.
        # vtx_weight = gate * mask_f (gate from Stage 1 origin probs).

        # ================================================================
        # Stage 3 — jet-flavour classification
        # ================================================================
        if tagging_feat_indices is None:
            tagging_feat_indices = list(range(in_dim))
        self.register_buffer("tagging_feat_idx",
            torch.tensor(tagging_feat_indices, dtype=torch.long))
        _stage3_in_dim = len(tagging_feat_indices)
        if stage3_use_origin_probs:
            _stage3_in_dim += n_origin_classes
        if stage3_use_vtx_weight:
            _stage3_in_dim += self.n_vertex_legs
        self.input_proj3 = nn.Linear(_stage3_in_dim, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if stage3_use_vertex_tokens:
            self.vertex_embed = nn.Sequential(
                nn.Linear(n_vtx_coords, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))

        _enc3_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder3 = nn.TransformerEncoder(_enc3_layer, num_layers=n_layers)
        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

        self.register_buffer("vertex_leg_origin_matrix",
            torch.tensor(vertex_leg_origin_matrix, dtype=torch.float32))
        self.d0_idx, self.d0_unc_idx, self.dphi_idx = d0_idx, d0_unc_idx, dphi_idx
        self.z0st_idx, self.z0st_unc_idx = z0st_idx, z0st_unc_idx
        self.theta_idx = theta_idx

    # ====================================================================
    # _vertex_fit_two_step
    # ====================================================================
    def _vertex_fit_two_step(self, x, vtx_weight):
        B = x.shape[0]
        L = self.n_vertex_legs

        if self.fit_lxy:
            d0      = x[..., self.d0_idx].unsqueeze(-1)
            d0_unc  = x[..., self.d0_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            dphi    = x[..., self.dphi_idx].unsqueeze(-1)
            sin_dphi   = torch.sin(dphi)
            cos_dphi   = torch.cos(dphi)
            sum_sin    = (vtx_weight * sin_dphi).sum(1)
            sum_cos    = (vtx_weight * cos_dphi).sum(1)
            flight_phi = torch.atan2(sum_sin, sum_cos).unsqueeze(1)
            flight_phi = torch.nan_to_num(flight_phi, nan=0.0)
            delta_phi  = dphi - flight_phi
            sin_d      = torch.sin(delta_phi)
            inv_var_d0 = vtx_weight / d0_unc.pow(2)
            num_lxy    = (inv_var_d0 * sin_d.abs() * d0.abs()).sum(1)
            den_lxy    = (inv_var_d0 * sin_d.pow(2)).sum(1).clamp(min=1e-6)
            lxy_pred   = num_lxy / den_lxy
            if self.calibrate_vertex_fit:
                lxy_pred = lxy_pred * torch.exp(self.lxy_log_scale).unsqueeze(0)
        else:
            lxy_pred   = vtx_weight.new_zeros(B, L)
            delta_phi  = None
            flight_phi = None

        if self.fit_dz:
            theta   = x[..., self.theta_idx].unsqueeze(-1)
            sin_th  = torch.sin(theta)
            cos_th  = torch.cos(theta)
            z0st     = x[..., self.z0st_idx].unsqueeze(-1)
            z0st_unc = x[..., self.z0st_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)

            delta_phi_z = dphi - flight_phi
            residual = z0st + lxy_pred.unsqueeze(1) * cos_th * torch.cos(delta_phi_z)

            inv_var_z0 = vtx_weight / z0st_unc.pow(2)
            num_dz     = (inv_var_z0 * sin_th * residual).sum(1)
            den_dz     = (inv_var_z0 * sin_th.pow(2)).sum(1).clamp(min=1e-6)
            dz_pred    = num_dz / den_dz
            if self.calibrate_vertex_fit:
                dz_pred = dz_pred * torch.exp(self.dz_log_scale).unsqueeze(0)
        else:
            dz_pred = vtx_weight.new_zeros(B, L)

        return lxy_pred, dz_pred, flight_phi, delta_phi

    # ====================================================================
    # _vertex_fit_old_dz
    # ====================================================================
    def _vertex_fit_old_dz(self, x, vtx_weight):
        B = x.shape[0]
        L = self.n_vertex_legs

        if self.fit_lxy:
            d0      = x[..., self.d0_idx].unsqueeze(-1)
            d0_unc  = x[..., self.d0_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            dphi    = x[..., self.dphi_idx].unsqueeze(-1)
            sin_dphi   = torch.sin(dphi)
            cos_dphi   = torch.cos(dphi)
            sum_sin    = (vtx_weight * sin_dphi).sum(1)
            sum_cos    = (vtx_weight * cos_dphi).sum(1)
            flight_phi = torch.atan2(sum_sin, sum_cos).unsqueeze(1)
            flight_phi = torch.nan_to_num(flight_phi, nan=0.0)
            delta_phi  = dphi - flight_phi
            sin_d      = torch.sin(delta_phi)
            inv_var_d0 = vtx_weight / d0_unc.pow(2)
            num_lxy    = (inv_var_d0 * sin_d.abs() * d0.abs()).sum(1)
            den_lxy    = (inv_var_d0 * sin_d.pow(2)).sum(1).clamp(min=1e-6)
            lxy_pred   = num_lxy / den_lxy
            if self.calibrate_vertex_fit:
                lxy_pred = lxy_pred * torch.exp(self.lxy_log_scale).unsqueeze(0)
        else:
            lxy_pred   = vtx_weight.new_zeros(B, L)
            delta_phi  = None
            flight_phi = None

        if self.fit_dz:
            z0st     = x[..., self.z0st_idx].unsqueeze(-1)
            z0st_unc = x[..., self.z0st_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            inv_var_z0 = vtx_weight / z0st_unc.pow(2)
            num_dz     = (inv_var_z0 * z0st).sum(1)
            den_dz     = inv_var_z0.sum(1).clamp(min=1e-6)
            dz_pred    = num_dz / den_dz
            if self.calibrate_vertex_fit:
                dz_pred = dz_pred * torch.exp(self.dz_log_scale).unsqueeze(0)
        else:
            dz_pred = vtx_weight.new_zeros(B, L)

        return lxy_pred, dz_pred, flight_phi, delta_phi

    # ====================================================================
    # _vertex_fit_3dwls
    # ====================================================================
    def _vertex_fit_3dwls(self, x, vtx_weight):
        B, K, L = vtx_weight.shape
        device  = x.device

        dphi    = x[..., self.dphi_idx]
        d0      = x[..., self.d0_idx]
        d0_unc  = x[..., self.d0_unc_idx].abs().clamp(min=1e-3)
        z0st    = x[..., self.z0st_idx]
        z0st_unc= x[..., self.z0st_unc_idx].abs().clamp(min=1e-3)
        theta   = x[..., self.theta_idx]
        sin_th  = torch.sin(theta)
        cos_th  = torch.cos(theta)

        sin_phi = torch.sin(dphi)
        cos_phi = torch.cos(dphi)

        I3 = torch.eye(3, device=device).unsqueeze(0)

        lxy_pred = vtx_weight.new_zeros(B, L)
        dz_pred  = vtx_weight.new_zeros(B, L)
        flight_phi = vtx_weight.new_zeros(B, L)
        delta_phi  = vtx_weight.new_zeros(B, K, L)

        for leg in range(L):
            w = vtx_weight[:, :, leg]

            w_d = w / d0_unc.pow(2)
            sin_mask = (sin_th.abs() >= MIN_SIN_THETA).float()
            w_z = w / z0st_unc.pow(2) * sin_mask

            a0 = torch.stack([-sin_phi, cos_phi,
                              torch.zeros_like(sin_phi)], dim=-1)
            a1 = torch.stack([-cos_th * cos_phi, -cos_th * sin_phi,
                              sin_th], dim=-1)

            M = (torch.einsum('bk,bki,bkj->bij', w_d, a0, a0) +
                 torch.einsum('bk,bki,bkj->bij', w_z, a1, a1))
            b = (torch.einsum('bk,bki,bk->bi', w_d, a0, d0) +
                 torch.einsum('bk,bki,bk->bi', w_z, a1, z0st))

            v = torch.linalg.solve(M + self.vertex_fit_reg * I3, b)
            X, Y, Z = v[:, 0], v[:, 1], v[:, 2]

            if self.fit_lxy:
                lxy_pred[:, leg]  = torch.sqrt(X.pow(2) + Y.pow(2) + 1e-12)
                flight_phi[:, leg] = torch.atan2(Y, X)
                flight_phi[:, leg] = torch.nan_to_num(flight_phi[:, leg], nan=0.0)
                delta_phi[:, :, leg] = dphi - flight_phi[:, leg].unsqueeze(1)
            if self.fit_dz:
                dz_pred[:, leg] = Z

        if self.calibrate_vertex_fit and self.fit_lxy:
            lxy_pred = lxy_pred * torch.exp(self.lxy_log_scale).unsqueeze(0)
        if self.calibrate_vertex_fit and self.fit_dz:
            dz_pred = dz_pred * torch.exp(self.dz_log_scale).unsqueeze(0)

        return lxy_pred, dz_pred, flight_phi, delta_phi

    # ====================================================================
    # forward
    # ====================================================================
    def forward(self, x, mask):
        B, K, _ = x.shape
        track_padding_mask = ~mask
        mask_f = mask.unsqueeze(-1).float()

        # ---- Stage 1: track-origin prediction ----
        h1 = self.input_proj1(x)
        h1 = self.encoder1(h1, src_key_padding_mask=track_padding_mask)
        origin_logits = self.origin_head(h1)
        soft_probs = torch.softmax(origin_logits, dim=-1) * mask_f

        # ---- Stage 2: gate-only vertex weighting (no refine) ----
        leg_origin_probs = soft_probs @ self.vertex_leg_origin_matrix
        gate = torch.sigmoid((leg_origin_probs - 0.5) / self.gate_temp)
        refine = torch.zeros_like(gate)
        vtx_weight = gate * mask_f

        if self.vertex_fit_method == "wls_3d":
            lxy_pred, dz_pred, flight_phi, delta_phi = \
                self._vertex_fit_3dwls(x, vtx_weight)
        elif self.vertex_fit_method == "old_dz":
            lxy_pred, dz_pred, flight_phi, delta_phi = \
                self._vertex_fit_old_dz(x, vtx_weight)
        else:
            lxy_pred, dz_pred, flight_phi, delta_phi = \
                self._vertex_fit_two_step(x, vtx_weight)

        # ---- Stage 3: jet-flavour classification ----
        if self.stage3_use_vertex_tokens:
            _vtx_parts = []
            if self.fit_lxy:
                _vtx_parts.append(torch.log1p(lxy_pred.clamp(min=0)))
            if self.fit_dz:
                _vtx_parts.append(torch.log1p(dz_pred.abs()))
            vtx_summary = torch.stack(_vtx_parts, dim=-1)
            vtx_tokens = self.vertex_embed(vtx_summary)

        x_tag = x.index_select(-1, self.tagging_feat_idx)
        _h3_input_parts = [x_tag]
        if self.stage3_use_origin_probs:
            _h3_input_parts.append(soft_probs)
        if self.stage3_use_vtx_weight:
            _h3_input_parts.append(vtx_weight)
        h3_input  = torch.cat(_h3_input_parts, dim=-1) if len(_h3_input_parts) > 1 else x_tag
        h3_tracks = self.input_proj3(h3_input)

        cls_tok = self.cls_token.expand(B, -1, -1)
        h3_parts = [cls_tok]
        if self.stage3_use_vertex_tokens:
            h3_parts.append(vtx_tokens)
        h3_parts.append(h3_tracks)
        h3_in = torch.cat(h3_parts, dim=1)

        n_extra_tokens = 1 + (self.n_vertex_legs if self.stage3_use_vertex_tokens else 0)
        extra_valid = torch.ones(B, n_extra_tokens, dtype=torch.bool,
                                 device=x.device)
        src_key_padding_mask3 = ~torch.cat([extra_valid, mask], dim=1)

        h3 = self.encoder3(h3_in, src_key_padding_mask=src_key_padding_mask3)
        jet_logits = self.jet_head(h3[:, 0])

        return {
            "jet_logits":        jet_logits,
            "origin_logits":     origin_logits,
            "vtx_weight":        vtx_weight,
            "leg_origin_probs":  leg_origin_probs,
            "gate":              gate,
            "refine":            refine,
            "lxy_pred":          lxy_pred,
            "dz_pred":           dz_pred,
        }

    def calibration_scales(self):
        out = {}
        if self.calibrate_vertex_fit and self.fit_lxy:
            out["lxy"] = torch.exp(self.lxy_log_scale).detach().cpu().tolist()
        if self.calibrate_vertex_fit and self.fit_dz:
            out["dz"] = torch.exp(self.dz_log_scale).detach().cpu().tolist()
        return out


def build_staged_origin_vertex_jet_no_refine(config):
    return StagedOriginVertexJetTransformerNoRefine(
        in_dim=config.n_feats, d_model=config.d_model,
        n_heads=config.n_heads, n_layers=config.n_layers,
        d_ffn=config.d_ffn, dropout=config.dropout,
        n_origin_classes=config.n_origin_classes,
        n_jet_classes=config.n_jet_classes,
        vertex_feat_indices=config.vertex_feat_idx,
        d0_idx=config.d0_idx, d0_unc_idx=config.d0_unc_idx,
        dphi_idx=config.dphi_idx,
        z0st_idx=config.z0st_idx, z0st_unc_idx=config.z0st_unc_idx,
        theta_idx=config.theta_idx,
        vertex_leg_origin_matrix=config.vertex_leg_origin_matrix,
        gate_temp=config.gate_temp,
        vertex_fit_method=config.vertex_fit_method,
        vertex_fit_reg=config.vertex_fit_reg,
        fit_lxy=config.fit_lxy, fit_dz=config.fit_dz,
        stage3_use_origin_probs=config.stage3_use_origin_probs,
        stage3_use_vtx_weight=config.stage3_use_vtx_weight,
        stage3_use_vertex_tokens=config.stage3_use_vertex_tokens,
        tagging_feat_indices=config.tagging_feat_idx,
        calibrate_vertex_fit=config.calibrate_vertex_fit,
    )
