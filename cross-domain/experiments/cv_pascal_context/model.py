"""Unscaled author HRNet-W18/ST and MTI-Net, with passive feature hooks."""
from types import SimpleNamespace
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
from experiments.cv_pascal_context.source import ROOT, author_module, verify_source

CHANNELS = (18, 36, 72, 144)
TASK_OUTPUTS = {"semseg": 21, "human_parts": 7, "sal": 1, "normals": 3, "edge": 1}
AUX_TASKS = ("human_parts", "sal", "normals", "edge")
VARIANTS = ("single_task", "main_only_matched", "multi_task")


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class AuthorNetwork(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown upstream variant: {variant}")
        import yaml
        hrnet = author_module("models.seg_hrnet")
        configuration = yaml.safe_load((ROOT / "vendor/author/models/model_info/hrnet_w18.yml").read_text())
        backbone = hrnet.HighResolutionNet(configuration)
        # Construct identical backbone and identical semantic head before any
        # auxiliary layers, so paired ST/MT seeds share initial common tensors.
        semantic_head = hrnet.HighResolutionHead(list(CHANNELS), 21)
        self.variant = variant
        if variant == "single_task":
            self.network = author_module("models.models").SingleTaskModel(backbone, semantic_head, "semseg")
        else:
            heads = nn.ModuleDict({"semseg": semantic_head})
            for task in AUX_TASKS:
                heads[task] = hrnet.HighResolutionHead(list(CHANNELS), TASK_OUTPUTS[task])
            tasks = SimpleNamespace(NAMES=list(TASK_OUTPUTS), NUM_OUTPUT=TASK_OUTPUTS)
            p = SimpleNamespace(TASKS=tasks, AUXILARY_TASKS=tasks)
            self.network = author_module("models.mti_net").MTINet(p, backbone, list(CHANNELS), heads)
        self._features = {}
        self.network.backbone.register_forward_hook(self._capture_backbone)
        final_head = self.network.decoder if variant == "single_task" else self.network.heads["semseg"]
        final_head.register_forward_hook(self._capture_native)
        if variant != "single_task":
            for i in range(4):
                getattr(self.network, f"distillation_scale_{i}").register_forward_hook(self._capture_distilled(i))

    def _capture_backbone(self, module, args, output):
        self._features["backbone"] = output

    def _capture_native(self, module, args, output):
        self._features["native_lowres"] = output

    def _capture_distilled(self, scale):
        def capture(module, args, output):
            self._features[f"distilled_{scale}"] = output
        return capture

    def forward(self, image):
        self._features = {}
        output = self.network(image)
        backbone = self._features["backbone"]
        output["native_lowres"] = self._features["native_lowres"]
        semantic = backbone if self.variant == "single_task" else [self._features[f"distilled_{i}"]["semseg"] for i in range(4)]
        output["h"] = [torch.cat((backbone[i], semantic[i]), 1) for i in range(4)]
        if self.variant != "single_task":
            output["aux_hidden"] = [torch.cat([self._features[f"distilled_{i}"][t] for t in AUX_TASKS], 1) for i in range(4)]
            # Both final predictions and all four initial scale predictions are
            # cached. No ground-truth-derived masks touch these tensors.
            output["aux_prediction"] = []
            for i in range(4):
                low = output["deep_supervision"][f"scale_{i}"]
                initial = [prediction_representation(low[t], t) for t in AUX_TASKS]
                final = [F.interpolate(prediction_representation(output[t], t), size=backbone[i].shape[-2:], mode="bilinear", align_corners=False) for t in AUX_TASKS]
                output["aux_prediction"].append(torch.cat(initial + final, 1))
        return output


def prediction_representation(logits, task):
    if task == "human_parts":
        return logits.softmax(1)
    if task == "normals":
        return F.normalize(logits, dim=1, eps=1e-12)
    return logits.sigmoid()


def pretrained_identity(context, required=True):
    from pipeline.io import sha256_file
    settings = context.config["model"]["pretrained"]
    path = context.resolve(settings["path"])
    expected = settings.get("sha256")
    if not path.is_file():
        if not required:
            return {"path": str(path), "available": False}
        raise FileNotFoundError(f"provide the author's HRNet-W18-small-v2 ImageNet weights locally: {path}")
    observed = sha256_file(path)
    if not expected or len(expected) != 64:
        raise ValueError("set model.pretrained.sha256 to the reviewed local checkpoint SHA256 before training")
    if observed != expected:
        raise ValueError("pretrained weight SHA256 mismatch")
    return {"path": str(path), "sha256": observed, "source": settings["source"], "exposure": settings["exposure"]}


def build_model(context, variant, load_pretrained=True):
    model = AuthorNetwork(variant)
    source = verify_source()
    if load_pretrained:
        provenance = pretrained_identity(context)
        checkpoint = torch.load(provenance["path"], map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        current = model.network.backbone.state_dict()
        accepted = {key: value for key, value in state.items() if key in current and value.shape == current[key].shape}
        missing = sorted(set(current) - set(accepted))
        # ImageNet classification-only heads may be omitted, encoder parameters
        # may not. num_batches_tracked may be absent in old published weights.
        if any(not key.endswith("num_batches_tracked") for key in missing):
            raise ValueError(f"pretrained encoder coverage incomplete: {missing[:8]}")
        current.update(accepted)
        model.network.backbone.load_state_dict(current)
        provenance["loaded_tensor_count"] = len(accepted)
    else:
        provenance = {"loaded": False, "reason": "loading experiment checkpoint or explicit offline contract"}
    return model, {"source": source, "pretrained": provenance,
                   "initial_backbone_sha256": state_digest(model.network.backbone.state_dict()),
                   "initial_semantic_head_sha256": state_digest((model.network.decoder if variant == "single_task" else model.network.heads["semseg"]).state_dict()),
                   "initial_model_sha256": state_digest(model.state_dict())}


def native_logits(batch):
    """Reproduce the author final bilinear upsample on the inference backend.

    Cache pre-upsample FP32 logits, never a quantized approximation. Output
    sizes come from RGB geometry, not from any label values or truth mask.
    """
    sizes = batch["output_size"]
    if sizes.ndim != 2 or sizes.shape[1] != 2 or not torch.equal(sizes, sizes[:1].expand_as(sizes)):
        raise ValueError("native reconstruction requires equal positive RGB output sizes in a batch")
    size = tuple(int(value) for value in sizes[0].tolist())
    if min(size) <= 0:
        raise ValueError("invalid RGB output size")
    return F.interpolate(batch["native_lowres"], size=size, mode="bilinear", align_corners=False)


class SpatialResidual(nn.Module):
    """Identical capacity/receptive field across recipes; exact native epoch 0."""
    def __init__(self, width=64):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(6*c, width, 1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU()) for c in CHANNELS])
        self.fusion = nn.Sequential(nn.Conv2d(4*width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 21, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, batch):
        size = batch["features"][0].shape[-2:]
        encoded = [F.interpolate(layer(value), size=size, mode="bilinear", align_corners=False) for layer, value in zip(self.projections, batch["features"])]
        residual = self.fusion(torch.cat(encoded, 1))
        native = native_logits(batch)
        return native + F.interpolate(residual, size=native.shape[-2:], mode="bilinear", align_corners=False)
