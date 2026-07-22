"""Two-pass 3D WLS staged model with a small residual reweighting MLP.

Stage 1 predicts track-origin probabilities.  Those probabilities define an
initial per-leg gate for a first analytic 3D WLS fit.  Standardised geometric
residuals relative to that fit are then processed by a per-track MLP which can
only make a bounded multiplicative correction to the gate.  A second analytic
3D WLS fit supplies the supervised vertex and the Stage-3 vertex token.

There is intentionally no Stage-2 Transformer: the learned component decides
which tracks are inconsistent with the initial common vertex, while all vertex
geometry remains in the closed-form fitter.
"""

import torch
import torch.nn as nn


MIN_SIN_THETA = 0.1
WEIGHT_EPS = 1e-8


class StagedOriginVertexJetResidualRefine(nn.Module):
    def __init__(
            self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
            n_origin_classes, n_jet_classes,
            d0_idx, d0_unc_idx, dphi_idx, z0st_idx, z0st_unc_idx,
            theta_idx, qoverp_idx, vertex_leg_origin_matrix,
            tagging_feat_indices, vertex_fit_reg=1e-6,
            fit_lxy=True, fit_dz=True, calibrate_vertex_fit=True,
            residual_refine_inputs=("geometry", "origin_probs"),
            residual_refine_hidden_dims=(32, 16),
            residual_refine_alpha=1.0, residual_vertex_detach=True,
            stage3_use_origin_probs=False, stage3_use_vtx_weight=False,
            stage3_use_vertex_tokens=True):
        super().__init__()
        self.n_origin_classes = n_origin_classes
        self.n_vertex_legs = vertex_leg_origin_matrix.shape[1]
        self.vertex_fit_reg = vertex_fit_reg
        self.fit_lxy = fit_lxy
        self.fit_dz = fit_dz
        self.calibrate_vertex_fit = calibrate_vertex_fit
        self.residual_refine_inputs = tuple(residual_refine_inputs)
        self.residual_refine_alpha = float(residual_refine_alpha)
        self.residual_vertex_detach = residual_vertex_detach
        self.stage3_use_origin_probs = stage3_use_origin_probs
        self.stage3_use_vtx_weight = stage3_use_vtx_weight
        self.stage3_use_vertex_tokens = stage3_use_vertex_tokens

        self.d0_idx = d0_idx
        self.d0_unc_idx = d0_unc_idx
        self.dphi_idx = dphi_idx
        self.z0st_idx = z0st_idx
        self.z0st_unc_idx = z0st_unc_idx
        self.theta_idx = theta_idx
        self.qoverp_idx = qoverp_idx

        self.register_buffer(
            "vertex_leg_origin_matrix",
            torch.tensor(vertex_leg_origin_matrix, dtype=torch.float32))
        self.register_buffer(
            "tagging_feat_idx",
            torch.tensor(tagging_feat_indices, dtype=torch.long))

        # Stage 1: unchanged origin-prediction Transformer.
        self.input_proj1 = nn.Linear(in_dim, d_model)
        encoder1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder1_layer, num_layers=n_layers)
        self.origin_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_origin_classes))

        # Stage 2: bounded residual MLP; no Transformer attention.
        residual_input_dim = 4 * self.n_vertex_legs
        if "origin_probs" in self.residual_refine_inputs:
            residual_input_dim += n_origin_classes
        if "qoverp" in self.residual_refine_inputs:
            if qoverp_idx < 0:
                raise ValueError("residual_refine_inputs includes qoverp but qOverP is absent")
            residual_input_dim += 1
        hidden1, hidden2 = residual_refine_hidden_dims
        self.residual_input_dim = residual_input_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, hidden1), nn.GELU(),
            nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Linear(hidden2, self.n_vertex_legs))
        # Start from the analytic gate (delta=0) instead of imposing a random
        # correction before the residual network has learned anything.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # Optional per-leg calibration, matching the existing staged model.
        if calibrate_vertex_fit and fit_lxy:
            self.lxy_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))
        if calibrate_vertex_fit and fit_dz:
            self.dz_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))

        # Stage 3: unchanged track/vertex-token jet classifier.
        stage3_input_dim = len(tagging_feat_indices)
        if stage3_use_origin_probs:
            stage3_input_dim += n_origin_classes
        if stage3_use_vtx_weight:
            stage3_input_dim += self.n_vertex_legs
        self.input_proj3 = nn.Linear(stage3_input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        n_vtx_coords = int(fit_lxy) + int(fit_dz)
        if stage3_use_vertex_tokens:
            self.vertex_embed = nn.Sequential(
                nn.Linear(n_vtx_coords, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))

        encoder3_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder3 = nn.TransformerEncoder(encoder3_layer, num_layers=n_layers)
        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

    def _geometry(self, x):
        """Return measurements, uncertainties and the two WLS design rows."""
        dphi = x[..., self.dphi_idx]
        d0 = x[..., self.d0_idx]
        d0_unc = x[..., self.d0_unc_idx].abs().clamp(min=1e-3)
        z0st = x[..., self.z0st_idx]
        z0st_unc = x[..., self.z0st_unc_idx].abs().clamp(min=1e-3)
        theta = x[..., self.theta_idx]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(dphi)
        cos_phi = torch.cos(dphi)

        transverse_row = torch.stack(
            [-sin_phi, cos_phi, torch.zeros_like(sin_phi)], dim=-1)
        longitudinal_row = torch.stack(
            [-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta], dim=-1)
        longitudinal_mask = (sin_theta.abs() >= MIN_SIN_THETA).to(x.dtype)
        return (d0, d0_unc, z0st, z0st_unc, transverse_row,
                longitudinal_row, longitudinal_mask)

    def _fit_3d_wls(self, x, weights, geometry=None):
        """Solve one analytic (X,Y,Z) fit for every batch item and leg."""
        if geometry is None:
            geometry = self._geometry(x)
        d0, d0_unc, z0st, z0st_unc, a_d0, a_z, z_mask = geometry
        batch_size = x.shape[0]
        identity = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        vertices = []

        for leg in range(self.n_vertex_legs):
            leg_weight = weights[:, :, leg]
            weight_d0 = leg_weight / d0_unc.square()
            weight_z = leg_weight / z0st_unc.square() * z_mask
            matrix = (
                torch.einsum("bk,bki,bkj->bij", weight_d0, a_d0, a_d0)
                + torch.einsum("bk,bki,bkj->bij", weight_z, a_z, a_z))
            rhs = (
                torch.einsum("bk,bki,bk->bi", weight_d0, a_d0, d0)
                + torch.einsum("bk,bki,bk->bi", weight_z, a_z, z0st))
            vertex = torch.linalg.solve(
                matrix + self.vertex_fit_reg * identity.expand(batch_size, -1, -1),
                rhs)
            vertices.append(vertex)

        vertex_xyz = torch.stack(vertices, dim=1)
        lxy = torch.sqrt(vertex_xyz[..., 0].square()
                         + vertex_xyz[..., 1].square() + 1e-12)
        dz = vertex_xyz[..., 2]
        return vertex_xyz, lxy, dz

    def _standardised_residuals(self, vertex_xyz, geometry):
        d0, d0_unc, z0st, z0st_unc, a_d0, a_z, z_mask = geometry
        predicted_d0 = torch.einsum("bki,bli->bkl", a_d0, vertex_xyz)
        predicted_z = torch.einsum("bki,bli->bkl", a_z, vertex_xyz)
        residual_d0 = (d0.unsqueeze(-1) - predicted_d0) / d0_unc.unsqueeze(-1)
        residual_z = ((z0st.unsqueeze(-1) - predicted_z)
                      / z0st_unc.unsqueeze(-1)) * z_mask.unsqueeze(-1)
        residual_d0 = torch.nan_to_num(residual_d0)
        residual_z = torch.nan_to_num(residual_z)
        return residual_d0, residual_z

    def _apply_calibration(self, lxy, dz):
        if self.calibrate_vertex_fit and self.fit_lxy:
            lxy = lxy * torch.exp(self.lxy_log_scale).unsqueeze(0)
        if self.calibrate_vertex_fit and self.fit_dz:
            dz = dz * torch.exp(self.dz_log_scale).unsqueeze(0)
        return lxy, dz

    def forward(self, x, mask):
        batch_size = x.shape[0]
        track_padding_mask = ~mask
        mask_float = mask.unsqueeze(-1).to(x.dtype)

        # Stage 1: origin probabilities define the initial per-leg gate.
        h1 = self.input_proj1(x)
        h1 = self.encoder1(h1, src_key_padding_mask=track_padding_mask)
        origin_logits = self.origin_head(h1)
        origin_probs = torch.softmax(origin_logits, dim=-1) * mask_float
        initial_gate = (origin_probs @ self.vertex_leg_origin_matrix) * mask_float

        # First analytic fit and detached standardised residuals.
        geometry = self._geometry(x)
        vertex0, initial_lxy, initial_dz = self._fit_3d_wls(
            x, initial_gate, geometry)
        residual_vertex = vertex0.detach() if self.residual_vertex_detach else vertex0
        residual_d0, residual_z = self._standardised_residuals(
            residual_vertex, geometry)
        residual_chi2 = residual_d0.square() + residual_z.square()

        # Geometry-only input is 4*L: [r_d0, r_z, r2, gate] for every leg.
        geometry_features = torch.stack(
            [residual_d0, residual_z, residual_chi2, initial_gate], dim=-1)
        mlp_parts = [geometry_features.flatten(start_dim=-2)]
        if "origin_probs" in self.residual_refine_inputs:
            mlp_parts.append(origin_probs)
        if "qoverp" in self.residual_refine_inputs:
            mlp_parts.append(x[..., self.qoverp_idx].unsqueeze(-1))
        residual_input = torch.cat(mlp_parts, dim=-1)

        delta = self.residual_refine_alpha * torch.tanh(
            self.residual_mlp(residual_input))
        relative_weight = torch.exp(delta)
        unnormalised_weight = initial_gate * relative_weight * mask_float
        vtx_weight = unnormalised_weight / (
            unnormalised_weight.sum(dim=1, keepdim=True) + WEIGHT_EPS)

        # Second analytic fit supplies the final vertex prediction.
        vertex1, lxy_pred, dz_pred = self._fit_3d_wls(x, vtx_weight, geometry)
        lxy_pred, dz_pred = self._apply_calibration(lxy_pred, dz_pred)

        # Stage 3: same configurable bridges and vertex tokens as staged.
        x_tag = x.index_select(-1, self.tagging_feat_idx)
        stage3_parts = [x_tag]
        if self.stage3_use_origin_probs:
            stage3_parts.append(origin_probs)
        if self.stage3_use_vtx_weight:
            stage3_parts.append(vtx_weight)
        stage3_input = (torch.cat(stage3_parts, dim=-1)
                        if len(stage3_parts) > 1 else x_tag)
        track_tokens = self.input_proj3(stage3_input)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sequence_parts = [cls_token]
        if self.stage3_use_vertex_tokens:
            vertex_parts = []
            if self.fit_lxy:
                vertex_parts.append(torch.log1p(lxy_pred.clamp(min=0)))
            if self.fit_dz:
                vertex_parts.append(torch.log1p(dz_pred.abs()))
            vertex_summary = torch.stack(vertex_parts, dim=-1)
            sequence_parts.append(self.vertex_embed(vertex_summary))
        sequence_parts.append(track_tokens)
        stage3_sequence = torch.cat(sequence_parts, dim=1)

        n_extra = 1 + (self.n_vertex_legs if self.stage3_use_vertex_tokens else 0)
        extra_valid = torch.ones(
            batch_size, n_extra, dtype=torch.bool, device=x.device)
        stage3_padding_mask = ~torch.cat([extra_valid, mask], dim=1)
        h3 = self.encoder3(
            stage3_sequence, src_key_padding_mask=stage3_padding_mask)
        jet_logits = self.jet_head(h3[:, 0])

        return {
            "jet_logits": jet_logits,
            "origin_logits": origin_logits,
            "vtx_weight": vtx_weight,
            "leg_origin_probs": initial_gate,
            "gate": initial_gate,
            "refine": relative_weight,
            "residual_delta": delta,
            "residual_d0": residual_d0,
            "residual_z": residual_z,
            "initial_vertex": vertex0,
            "initial_lxy_pred": initial_lxy,
            "initial_dz_pred": initial_dz,
            "vertex": vertex1,
            "lxy_pred": lxy_pred,
            "dz_pred": dz_pred,
        }

    def calibration_scales(self):
        scales = {}
        if self.calibrate_vertex_fit and self.fit_lxy:
            scales["lxy"] = torch.exp(self.lxy_log_scale).detach().cpu().tolist()
        if self.calibrate_vertex_fit and self.fit_dz:
            scales["dz"] = torch.exp(self.dz_log_scale).detach().cpu().tolist()
        return scales


def build_staged_origin_vertex_jet_residual_refine(config):
    if config.vertex_fit_method != "wls_3d":
        raise ValueError(
            "staged_origin_vertex_jet_residual_refine requires vertex_fit_method='wls_3d'")
    if config.theta_idx < 0:
        raise ValueError(
            "staged_origin_vertex_jet_residual_refine requires the theta track field")
    return StagedOriginVertexJetResidualRefine(
        in_dim=config.n_feats,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        n_origin_classes=config.n_origin_classes,
        n_jet_classes=config.n_jet_classes,
        d0_idx=config.d0_idx,
        d0_unc_idx=config.d0_unc_idx,
        dphi_idx=config.dphi_idx,
        z0st_idx=config.z0st_idx,
        z0st_unc_idx=config.z0st_unc_idx,
        theta_idx=config.theta_idx,
        qoverp_idx=config.qoverp_idx,
        vertex_leg_origin_matrix=config.vertex_leg_origin_matrix,
        tagging_feat_indices=config.tagging_feat_idx,
        vertex_fit_reg=config.vertex_fit_reg,
        fit_lxy=config.fit_lxy,
        fit_dz=config.fit_dz,
        calibrate_vertex_fit=config.calibrate_vertex_fit,
        residual_refine_inputs=config.residual_refine_inputs,
        residual_refine_hidden_dims=config.residual_refine_hidden_dims,
        residual_refine_alpha=config.residual_refine_alpha,
        residual_vertex_detach=config.residual_vertex_detach,
        stage3_use_origin_probs=config.stage3_use_origin_probs,
        stage3_use_vtx_weight=config.stage3_use_vtx_weight,
        stage3_use_vertex_tokens=config.stage3_use_vertex_tokens,
    )
