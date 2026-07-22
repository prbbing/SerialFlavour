"""Origin-free model with a learned track-to-vertex assignment encoder.

The O2 model removes the complete origin encoder/head and replaces it with a
compact assignment Transformer operating only on selected geometry and
kinematic fields. It predicts b-leg, c-leg and other probabilities; the first
two probabilities are used directly as analytic 3D-WLS weights. Truth labels
are never accepted by the forward interface.
"""

import torch
import torch.nn as nn


MIN_SIN_THETA = 0.1


class StagedOriginVertexJetTrackAblation(nn.Module):
    def __init__(
            self, d_model, n_heads, n_layers, d_ffn, dropout,
            n_vertex_legs, n_jet_classes, assignment_feat_indices,
            assignment_n_layers,
            d0_idx, d0_unc_idx, dphi_idx, z0st_idx, z0st_unc_idx, theta_idx,
            tagging_feat_indices, vertex_fit_reg=1e-6,
            fit_lxy=True, fit_dz=True,
            calibrate_vertex_fit=True, stage3_use_vtx_weight=True,
            stage3_use_vertex_tokens=True, detach_vertex_from_jet=False):
        super().__init__()
        self.n_vertex_legs = n_vertex_legs
        self.n_assignment_classes = self.n_vertex_legs + 1
        self.vertex_fit_reg = vertex_fit_reg
        self.fit_lxy = fit_lxy
        self.fit_dz = fit_dz
        self.calibrate_vertex_fit = calibrate_vertex_fit
        self.stage3_use_vtx_weight = stage3_use_vtx_weight
        self.stage3_use_vertex_tokens = stage3_use_vertex_tokens
        self.detach_vertex_from_jet = detach_vertex_from_jet

        self.d0_idx = d0_idx
        self.d0_unc_idx = d0_unc_idx
        self.dphi_idx = dphi_idx
        self.z0st_idx = z0st_idx
        self.z0st_unc_idx = z0st_unc_idx
        self.theta_idx = theta_idx
        self.register_buffer(
            "assignment_feat_idx",
            torch.tensor(assignment_feat_indices, dtype=torch.long))
        self.register_buffer(
            "tagging_feat_idx",
            torch.tensor(tagging_feat_indices, dtype=torch.long))

        self.assignment_input_proj = nn.Linear(
            len(assignment_feat_indices), d_model)
        assignment_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.assignment_encoder = nn.TransformerEncoder(
            assignment_layer, num_layers=assignment_n_layers)
        self.assignment_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.n_assignment_classes))

        if calibrate_vertex_fit and fit_lxy:
            self.lxy_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))
        if calibrate_vertex_fit and fit_dz:
            self.dz_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))

        stage3_input_dim = len(tagging_feat_indices)
        if stage3_use_vtx_weight:
            stage3_input_dim += self.n_vertex_legs
        self.input_proj3 = nn.Linear(stage3_input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        n_vertex_coords = int(fit_lxy) + int(fit_dz)
        if stage3_use_vertex_tokens:
            self.vertex_embed = nn.Sequential(
                nn.Linear(n_vertex_coords, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))

        stage3_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder3 = nn.TransformerEncoder(stage3_layer, num_layers=n_layers)
        self.jet_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

    def _learned_assignment(self, x, mask):
        assignment_input = x.index_select(-1, self.assignment_feat_idx)
        hidden = self.assignment_input_proj(assignment_input)
        hidden = self.assignment_encoder(
            hidden, src_key_padding_mask=~mask)
        logits = self.assignment_head(hidden)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * mask.unsqueeze(-1).to(x.dtype)
        return logits, probabilities

    def _fit_3d_wls(self, x, weights):
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
        identity = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        vertices = []

        for leg in range(self.n_vertex_legs):
            leg_weight = weights[:, :, leg]
            transverse_weight = leg_weight / d0_unc.square()
            longitudinal_weight = (
                leg_weight / z0st_unc.square() * longitudinal_mask)
            matrix = (
                torch.einsum(
                    "bk,bki,bkj->bij", transverse_weight,
                    transverse_row, transverse_row)
                + torch.einsum(
                    "bk,bki,bkj->bij", longitudinal_weight,
                    longitudinal_row, longitudinal_row))
            rhs = (
                torch.einsum(
                    "bk,bki,bk->bi", transverse_weight, transverse_row, d0)
                + torch.einsum(
                    "bk,bki,bk->bi", longitudinal_weight,
                    longitudinal_row, z0st))
            vertex = torch.linalg.solve(
                matrix + self.vertex_fit_reg * identity, rhs)
            vertices.append(vertex)

        vertex_xyz = torch.stack(vertices, dim=1)
        lxy = torch.sqrt(
            vertex_xyz[..., 0].square() + vertex_xyz[..., 1].square() + 1e-12)
        dz = vertex_xyz[..., 2]
        if self.calibrate_vertex_fit and self.fit_lxy:
            lxy = lxy * torch.exp(self.lxy_log_scale).unsqueeze(0)
        if self.calibrate_vertex_fit and self.fit_dz:
            dz = dz * torch.exp(self.dz_log_scale).unsqueeze(0)
        return vertex_xyz, lxy, dz

    def forward(self, x, mask):
        batch_size = x.shape[0]
        mask_float = mask.unsqueeze(-1).to(x.dtype)
        assignment_logits, assignment_probs = self._learned_assignment(x, mask)

        # b/c probabilities are used directly as WLS weights; the final class
        # is the explicit non-vertex/other assignment.
        vtx_weight = assignment_probs[..., :self.n_vertex_legs] * mask_float
        vertex_xyz, lxy_pred, dz_pred = self._fit_3d_wls(x, vtx_weight)

        # O3a keeps the vertex objective trainable but prevents the jet
        # objective from updating the assignment encoder through either the
        # direct weight input or the fitted-vertex-token path.
        if self.detach_vertex_from_jet:
            stage3_vtx_weight = vtx_weight.detach()
            stage3_lxy = lxy_pred.detach()
            stage3_dz = dz_pred.detach()
        else:
            stage3_vtx_weight = vtx_weight
            stage3_lxy = lxy_pred
            stage3_dz = dz_pred

        tagging_input = x.index_select(-1, self.tagging_feat_idx)
        stage3_parts = [tagging_input]
        if self.stage3_use_vtx_weight:
            stage3_parts.append(stage3_vtx_weight)
        stage3_input = torch.cat(stage3_parts, dim=-1)
        track_tokens = self.input_proj3(stage3_input)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sequence_parts = [cls_token]
        if self.stage3_use_vertex_tokens:
            vertex_parts = []
            if self.fit_lxy:
                vertex_parts.append(torch.log1p(stage3_lxy.clamp(min=0)))
            if self.fit_dz:
                vertex_parts.append(torch.log1p(stage3_dz.abs()))
            vertex_summary = torch.stack(vertex_parts, dim=-1)
            sequence_parts.append(self.vertex_embed(vertex_summary))
        sequence_parts.append(track_tokens)
        sequence = torch.cat(sequence_parts, dim=1)

        n_extra_tokens = 1 + (
            self.n_vertex_legs if self.stage3_use_vertex_tokens else 0)
        extra_valid = torch.ones(
            batch_size, n_extra_tokens, dtype=torch.bool, device=x.device)
        stage3_padding_mask = ~torch.cat([extra_valid, mask], dim=1)
        hidden3 = self.encoder3(
            sequence, src_key_padding_mask=stage3_padding_mask)
        jet_logits = self.jet_head(hidden3[:, 0])

        output = {
            "jet_logits": jet_logits,
            "assignment_probs": assignment_probs,
            "vtx_weight": vtx_weight,
            "leg_origin_probs": vtx_weight,
            "gate": vtx_weight,
            "refine": torch.ones_like(vtx_weight) * mask_float,
            "vertex": vertex_xyz,
            "lxy_pred": lxy_pred,
            "dz_pred": dz_pred,
        }
        output["assignment_logits"] = assignment_logits
        return output

    def calibration_scales(self):
        scales = {}
        if self.calibrate_vertex_fit and self.fit_lxy:
            scales["lxy"] = torch.exp(self.lxy_log_scale).detach().cpu().tolist()
        if self.calibrate_vertex_fit and self.fit_dz:
            scales["dz"] = torch.exp(self.dz_log_scale).detach().cpu().tolist()
        return scales


def build_staged_origin_vertex_jet_track_ablation(config):
    if config.vertex_fit_method != "wls_3d":
        raise ValueError(
            "track-ablation model requires vertex_fit_method='wls_3d'")
    if config.stage3_use_origin_probs:
        raise ValueError(
            "origin-free models cannot include origin_probs in stage3_extra_inputs")
    return StagedOriginVertexJetTrackAblation(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        dropout=config.dropout,
        n_vertex_legs=config.n_vertex_legs,
        n_jet_classes=config.n_jet_classes,
        assignment_feat_indices=config.track_assignment_feat_idx,
        assignment_n_layers=config.track_assignment_n_layers,
        d0_idx=config.d0_idx,
        d0_unc_idx=config.d0_unc_idx,
        dphi_idx=config.dphi_idx,
        z0st_idx=config.z0st_idx,
        z0st_unc_idx=config.z0st_unc_idx,
        theta_idx=config.theta_idx,
        tagging_feat_indices=config.tagging_feat_idx,
        vertex_fit_reg=config.vertex_fit_reg,
        fit_lxy=config.fit_lxy,
        fit_dz=config.fit_dz,
        calibrate_vertex_fit=config.calibrate_vertex_fit,
        stage3_use_vtx_weight=config.stage3_use_vtx_weight,
        stage3_use_vertex_tokens=config.stage3_use_vertex_tokens,
        detach_vertex_from_jet=config.detach_vertex_from_jet,
    )
