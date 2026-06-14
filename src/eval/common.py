"""
eval.common
=====================
Shared plumbing for the faithfulness and localization evaluations.

Both evals load the same kinds of per-layer lens checkpoints and turn lens output into
class logits the same way; only the metric on top differs. This module holds everything
they share: the per-layer lens model, the disk-layout inspection + weight loaders, the
``make_logits_fn`` application path, preprocessing, forward hooks, and the batched
insertion/deletion curve helpers.

Geometry constants below are fixed for the CLIP ViT-L/14 backbone this project targets.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

# --- model geometry (CLIP ViT-L/14 @ 224) ---
IMG_SIZE = 224
PATCH_SIZE = 14
GRID = IMG_SIZE // PATCH_SIZE          # 16
IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)
NUM_LAYERS = 24
EMBED_DIM = 1024
UBLUR_RADIUS = 10                      # insertion baseline GaussianBlur radius

CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2758]

DEFAULT_MODEL_NAME = "vit_large_patch14_clip_224.openai_ft_in1k"


# ============================================================
# MODELS
# ============================================================
class LowRankMapper(nn.Module):
    """LowRankTransformation — saved keys: down.weight, up.weight, bias."""
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.up(self.down(x)) + self.bias


class ProjectionBlock(nn.Module):
    """Two-stage block: mapper -> lens. Used for mapper_type in {full, low_rank}.

    Lens is either a single Linear (affine: linear.weight) or a small MLP
    (mlp_kld: net.0.*, net.2.* under a Sequential named ``net`` in training).
    """
    def __init__(self, dim, mapper_type, mapper_rank, lens_out_dim, lens_mlp_hidden=None):
        super().__init__()
        if mapper_type == "full":
            self.mapper = nn.Linear(dim, dim)
        elif mapper_type == "low_rank":
            self.mapper = LowRankMapper(dim, dim, mapper_rank)
        else:
            raise ValueError(f"Unknown mapper_type={mapper_type}")
        if lens_mlp_hidden is None:
            self.lens = nn.Linear(dim, lens_out_dim)
        else:
            self.lens = nn.Sequential(
                nn.Linear(dim, lens_mlp_hidden),
                nn.GELU(),
                nn.Linear(lens_mlp_hidden, lens_out_dim),
            )

    def forward(self, x):
        return self.lens(self.mapper(x))


class UnifiedBlock(nn.Module):
    """Single Linear(in_dim, out_dim). For 'unified', the classifier comes from the
    frozen ViT (norm + fc_norm + head). out_dim defaults to in_dim."""
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.lens = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lens(x)


class MultiLayerLens(nn.Module):
    """Holds one block per layer; type depends on mapper_type.

    For mapper_type == 'direct_lens', only ONE block is built (for `direct_lens_layer`).
    """
    def __init__(self, num_layers, dim, mapper_type, per_layer_ranks=None, lens_out_dim=None,
                 direct_lens_layer=None, lens_mlp_hidden=None):
        super().__init__()
        if mapper_type == "unified":
            out_dim = dim if lens_out_dim is None else lens_out_dim
            self.projections = nn.ModuleDict({
                f"layer_{i}": UnifiedBlock(dim, out_dim) for i in range(num_layers)
            })
        elif mapper_type == "direct_lens":
            assert direct_lens_layer is not None and lens_out_dim is not None, \
                "direct_lens_layer and lens_out_dim are required for direct_lens."
            self.projections = nn.ModuleDict({
                direct_lens_layer: UnifiedBlock(dim, lens_out_dim)
            })
        else:
            assert per_layer_ranks is not None and lens_out_dim is not None, \
                "per_layer_ranks and lens_out_dim are required for full/low_rank."
            self.projections = nn.ModuleDict({
                f"layer_{i}": ProjectionBlock(
                    dim, mapper_type, per_layer_ranks[i], lens_out_dim,
                    lens_mlp_hidden=lens_mlp_hidden,
                )
                for i in range(num_layers)
            })


# ============================================================
# WEIGHT LOADER
# ============================================================
def _unwrap(sd):
    """Unwrap common state_dict wrappers."""
    if isinstance(sd, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                return sd[key]
    return sd


def _resolve_unified_path(weights_dir, layer_idx):
    """Try common naming conventions for unified-lens checkpoints."""
    for name in (f"layer{layer_idx}_best.pt",
                 f"layer_{layer_idx}.pt",
                 f"layer_{layer_idx}_best.pt",
                 f"layer_{layer_idx:02d}_best.pt",
                 f"layer{layer_idx}.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            return p
    return None


def _read_direct_lens_weight_keys(sd):
    """Find the (weight_key, bias_key) pair inside an unwrapped direct-lens state_dict."""
    if "proj.weight" in sd and "proj.bias" in sd:
        return "proj.weight", "proj.bias"
    if "weight" in sd and "bias" in sd:
        return "weight", "bias"
    if "linear.weight" in sd and "linear.bias" in sd:
        return "linear.weight", "linear.bias"
    raise KeyError(
        f"Direct lens checkpoint has unexpected keys: {list(sd.keys())}. "
        f"Expected (proj.weight, proj.bias) or (weight, bias) or (linear.weight, linear.bias)."
    )


def _infer_affine_lens_out_dim(l_sd):
    return int(l_sd["linear.weight"].shape[0])


def _infer_mlp_lens_dims(l_sd):
    """mlp_kld saves a Sequential under ``net`` -> net.0.weight, net.2.weight."""
    if "net.0.weight" not in l_sd or "net.2.weight" not in l_sd:
        raise KeyError(
            "Expected MLP lens keys 'net.0.weight' and 'net.2.weight'. "
            f"Got keys={list(l_sd.keys())}"
        )
    hidden = int(l_sd["net.0.weight"].shape[0])
    out_dim = int(l_sd["net.2.weight"].shape[0])
    return hidden, out_dim


def _infer_lens_out_and_mlp_hidden(l_sd, layer_idx_for_error):
    """Returns (lens_out_dim, lens_mlp_hidden); lens_mlp_hidden is None for affine lenses."""
    if "linear.weight" in l_sd:
        return _infer_affine_lens_out_dim(l_sd), None
    if any(k.startswith("net.") for k in l_sd):
        h, out_d = _infer_mlp_lens_dims(l_sd)
        return out_d, h
    raise KeyError(
        f"Layer {layer_idx_for_error} lens file: expected 'linear.weight' (affine) or "
        f"'net.*' (MLP). Got keys={list(l_sd.keys())}"
    )


def inspect_disk_layout(mapper_dir, lens_dir, num_layers, mapper_type, device,
                        unified_dir=None, direct_lens_path=None):
    """Returns (per_layer_ranks, detected_lens_out_dim, lens_mlp_hidden).

      unified     : dim is fixed (1024 -> 1024); classifier is the ViT's head.
      direct_lens : reads the single checkpoint and infers out_dim from proj.weight.
      full/low_rank : ranks (low_rank) and lens out dim are inferred from disk.
    """
    if mapper_type == "unified":
        return [0] * num_layers, EMBED_DIM, None

    if mapper_type == "direct_lens":
        if direct_lens_path is None:
            raise ValueError("direct_lens_path is required for mapper_type='direct_lens'.")
        if not os.path.exists(direct_lens_path):
            raise FileNotFoundError(f"Direct lens checkpoint not found: {direct_lens_path}")
        ckpt = torch.load(direct_lens_path, map_location=device)
        ckpt_type = ckpt.get("type", None) if isinstance(ckpt, dict) else None
        sd = _unwrap(ckpt)
        w_key, _ = _read_direct_lens_weight_keys(sd)
        out_dim = int(sd[w_key].shape[0])
        if ckpt_type is not None:
            print(f"  [direct_lens] checkpoint type='{ckpt_type}', detected out_dim={out_dim}")
        return [0] * num_layers, out_dim, None

    per_layer_ranks = []
    detected_lens_out = None
    lens_mlp_hidden = None

    for i in range(num_layers):
        m_path = os.path.join(mapper_dir, f"layer_{i}.pt")
        l_path = os.path.join(lens_dir, f"layer_{i}.pt")
        if not os.path.exists(m_path) or not os.path.exists(l_path):
            per_layer_ranks.append(0)
            continue

        m_sd = _unwrap(torch.load(m_path, map_location=device))
        l_sd = _unwrap(torch.load(l_path, map_location=device))

        if mapper_type == "low_rank":
            if "down.weight" not in m_sd:
                raise KeyError(
                    f"Layer {i} mapper file does not contain 'down.weight'. "
                    f"Got keys={list(m_sd.keys())}. Did you mean mapper_type='full'?"
                )
            per_layer_ranks.append(int(m_sd["down.weight"].shape[0]))
        else:
            per_layer_ranks.append(0)

        if detected_lens_out is None:
            out_d, mlp_h = _infer_lens_out_and_mlp_hidden(l_sd, i)
            detected_lens_out = out_d
            lens_mlp_hidden = mlp_h
        else:
            out_d, mlp_h = _infer_lens_out_and_mlp_hidden(l_sd, i)
            if out_d != detected_lens_out:
                raise ValueError(
                    f"Lens out_dim mismatch at layer {i}: disk has {out_d}, "
                    f"expected {detected_lens_out} from earlier layer."
                )
            if mlp_h != lens_mlp_hidden:
                raise ValueError(
                    f"MLP lens hidden dim mismatch at layer {i}: disk has {mlp_h}, "
                    f"expected {lens_mlp_hidden} from earlier layer."
                )

    if detected_lens_out is None:
        raise RuntimeError("No lens files found to infer lens_out_dim from.")
    return per_layer_ranks, detected_lens_out, lens_mlp_hidden


def load_per_layer_weights(lens_model, mapper_dir, lens_dir, num_layers, mapper_type, device,
                           unified_dir=None, direct_lens_path=None, direct_lens_layer=None):
    """Map disk-keys -> module-keys for each mapper_type, then load (strict=False)."""
    combined = {}
    missing_files = []

    if mapper_type == "direct_lens":
        if direct_lens_path is None or direct_lens_layer is None:
            raise ValueError("direct_lens_path and direct_lens_layer are required "
                             "when mapper_type='direct_lens'.")
        if not os.path.exists(direct_lens_path):
            raise FileNotFoundError(f"Direct lens checkpoint not found: {direct_lens_path}")

        ckpt = torch.load(direct_lens_path, map_location=device)
        sd = _unwrap(ckpt)
        w_key, b_key = _read_direct_lens_weight_keys(sd)

        prefix = f"projections.{direct_lens_layer}.lens."
        combined[prefix + "weight"] = sd[w_key]
        combined[prefix + "bias"] = sd[b_key]

        missing, unexpected = lens_model.load_state_dict(combined, strict=False)
        print(f"Loaded direct lens '{direct_lens_path}' for {direct_lens_layer}. "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        return lens_model

    if mapper_type == "unified":
        if unified_dir is None:
            raise ValueError("unified_dir must be provided when mapper_type='unified'.")

        for i in range(num_layers):
            path = _resolve_unified_path(unified_dir, i)
            if path is None:
                missing_files.append(f"<unified layer_{i} not found in {unified_dir}>")
                continue
            sd = _unwrap(torch.load(path, map_location=device))

            if "weight" in sd and "bias" in sd:
                w_key, b_key = "weight", "bias"
            elif "linear.weight" in sd and "linear.bias" in sd:
                w_key, b_key = "linear.weight", "linear.bias"
            elif "proj.weight" in sd and "proj.bias" in sd:
                w_key, b_key = "proj.weight", "proj.bias"
            else:
                raise KeyError(
                    f"Unified layer {i} file at {path} has unexpected keys: "
                    f"{list(sd.keys())}. Expected (weight, bias) or (linear.weight, linear.bias)."
                )

            prefix = f"projections.layer_{i}.lens."
            combined[prefix + "weight"] = sd[w_key]
            combined[prefix + "bias"] = sd[b_key]

    else:
        for i in range(num_layers):
            m_path = os.path.join(mapper_dir, f"layer_{i}.pt")
            l_path = os.path.join(lens_dir, f"layer_{i}.pt")
            if not os.path.exists(m_path):
                missing_files.append(m_path); continue
            if not os.path.exists(l_path):
                missing_files.append(l_path); continue

            m_sd = _unwrap(torch.load(m_path, map_location=device))
            l_sd = _unwrap(torch.load(l_path, map_location=device))
            prefix = f"projections.layer_{i}."

            if mapper_type == "full":
                combined[prefix + "mapper.weight"] = m_sd["linear.weight"]
                combined[prefix + "mapper.bias"] = m_sd["linear.bias"]
            elif mapper_type == "low_rank":
                combined[prefix + "mapper.down.weight"] = m_sd["down.weight"]
                combined[prefix + "mapper.up.weight"] = m_sd["up.weight"]
                combined[prefix + "mapper.bias"] = m_sd["bias"]
            else:
                raise ValueError(f"Unknown mapper_type={mapper_type}")

            if "linear.weight" in l_sd:
                combined[prefix + "lens.weight"] = l_sd["linear.weight"]
                combined[prefix + "lens.bias"] = l_sd["linear.bias"]
            elif any(k.startswith("net.") for k in l_sd):
                for k, v in l_sd.items():
                    if k.startswith("net."):
                        combined[prefix + "lens." + k[len("net."):]] = v
            else:
                raise KeyError(
                    f"Layer {i} lens: expected affine (linear.weight) or MLP (net.*) keys. "
                    f"Got {list(l_sd.keys())}"
                )

    if missing_files:
        print(f"[warn] {len(missing_files)} per-layer files missing; first few: {missing_files[:3]}")

    missing, unexpected = lens_model.load_state_dict(combined, strict=False)
    print(f"Loaded per-layer weights. missing={len(missing)} unexpected={len(unexpected)}")
    return lens_model


# ============================================================
# Application path: turn lens output into class logits
# ============================================================
def make_logits_fn(clip_model, mapper_type, use_vit_head):
    """Build the (lens-output -> class logits) function for the given mapper_type.

    unified     : lens(D->D) -> ViT.norm + fc_norm + head.
    direct_lens : D->1024 -> ViT.fc_norm + head only; D->1000 -> already logits.
    full/low_rank : D->1024 -> ViT.head; D->1000 -> already logits.
    """
    if mapper_type == "unified":
        norm = clip_model.norm
        fc_norm = getattr(clip_model, "fc_norm", nn.Identity())
        head = clip_model.head
        for m in (norm, fc_norm, head):
            for p in m.parameters():
                p.requires_grad_(False)

        def apply(x):
            return head(fc_norm(norm(x)))
        return apply

    if mapper_type == "direct_lens" and use_vit_head:
        fc_norm = getattr(clip_model, "fc_norm", nn.Identity())
        head = clip_model.head
        for m in (fc_norm, head):
            for p in m.parameters():
                p.requires_grad_(False)

        def apply(x):
            return head(fc_norm(x))
        return apply

    if use_vit_head:
        head = clip_model.head
        return lambda x: head(x)

    return lambda x: x  # already logits


# ============================================================
# MODEL LOADING + PREPROCESSING
# ============================================================
def load_clip_model(model_name, weights_path, device):
    """Load the frozen timm backbone (local checkpoint if present, else pretrained)."""
    clip_model = timm.create_model(model_name, pretrained=not (weights_path and os.path.exists(weights_path)))
    if weights_path and os.path.exists(weights_path):
        clip_model.load_state_dict(torch.load(weights_path, map_location=device))
    clip_model = clip_model.to(device)
    clip_model.eval()
    return clip_model


preprocess = transforms.Compose([
    transforms.Resize(size=IMG_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def transform_image(pil_img, device):
    return preprocess(pil_img).unsqueeze(0).to(device)


def make_activation_hook(store, name):
    """Forward hook that stores a layer's detached output into ``store[name]``."""
    def hook(_m, _i, output):
        store[name] = output.detach()
    return hook


def resolve_image_path(image_dir, image_name):
    """Try as-is first, then with .JPEG appended."""
    p = os.path.join(image_dir, image_name)
    if os.path.exists(p):
        return p
    p2 = os.path.join(image_dir, f"{image_name}.JPEG")
    if os.path.exists(p2):
        return p2
    return None


# ============================================================
# MASKING & BATCHED INSERTION / DELETION CURVES
# ============================================================
def patch_coords(idx):
    r, c = idx // GRID, idx % GRID
    return r * PATCH_SIZE, (r + 1) * PATCH_SIZE, c * PATCH_SIZE, (c + 1) * PATCH_SIZE


def mask_patch_tensor_(img_tensor, idx):
    y0, y1, x0, x1 = patch_coords(int(idx))
    img_tensor[:, :, y0:y1, x0:x1] = 0.0
    return img_tensor


def insert_patch_tensor_(current, original, idx):
    y0, y1, x0, x1 = patch_coords(int(idx))
    current[:, :, y0:y1, x0:x1] = original[:, :, y0:y1, x0:x1]
    return current


@torch.no_grad()
def get_class_probability_batch(clip_model, batch, class_id):
    return F.softmax(clip_model(batch), dim=-1)[:, class_id]


@torch.no_grad()
def deletion_curve_batched(clip_model, pil_img, class_id, ranked, eval_batch, device):
    fractions, probabilities = [], []
    img_tensor = transform_image(pil_img, device)
    n, k = len(ranked), 0
    while k < n:
        states, ks = [], []
        for _ in range(eval_batch):
            if k >= n:
                break
            mask_patch_tensor_(img_tensor, int(ranked[k]))
            states.append(img_tensor.clone())
            ks.append(k); k += 1
        probs = get_class_probability_batch(clip_model, torch.cat(states, 0), class_id).cpu().tolist()
        for j, p in enumerate(probs):
            fractions.append((ks[j] + 1) / n); probabilities.append(float(p))
    return fractions, probabilities


@torch.no_grad()
def insertion_curve_ublur_batched(clip_model, pil_img, class_id, ranked, eval_batch, device):
    fractions, probabilities = [], []
    original = transform_image(pil_img, device)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=UBLUR_RADIUS))
    current = transform_image(blurred, device)
    n, k = len(ranked), 0
    while k < n:
        states, ks = [], []
        for _ in range(eval_batch):
            if k >= n:
                break
            insert_patch_tensor_(current, original, int(ranked[k]))
            states.append(current.clone())
            ks.append(k); k += 1
        probs = get_class_probability_batch(clip_model, torch.cat(states, 0), class_id).cpu().tolist()
        for j, p in enumerate(probs):
            fractions.append((ks[j] + 1) / n); probabilities.append(float(p))
    return fractions, probabilities


# ============================================================
# Shared lens-model build (used by both evals)
# ============================================================
def build_lens_model(clip_model, mapper_type, target_layers, device,
                     mapper_dir=None, lens_dir=None, unified_dir=None,
                     direct_lens_path=None, direct_lens_layer=None,
                     lens_out_dim_override=None):
    """Inspect disk, build a ``MultiLayerLens``, load per-layer weights, and build the
    logits function — the common setup both evals run before scoring.

    Returns ``(lens_model, logits_fn, effective_target_layers)``.
    """
    if mapper_type == "direct_lens":
        effective_layers = [direct_lens_layer]
        print(f"  [direct_lens] checkpoint = {direct_lens_path}")
        print(f"  [direct_lens] restricting target layers to: {effective_layers}")
    else:
        effective_layers = list(target_layers)

    per_layer_ranks, detected_lens_out, lens_mlp_hidden = inspect_disk_layout(
        mapper_dir, lens_dir, NUM_LAYERS, mapper_type, device,
        unified_dir=unified_dir, direct_lens_path=direct_lens_path,
    )
    print(f"  Detected lens out_dim from disk: {detected_lens_out}"
          + (f"  (MLP hidden={lens_mlp_hidden})" if lens_mlp_hidden is not None else ""))

    if mapper_type not in ("unified", "direct_lens") and lens_out_dim_override is not None \
            and lens_out_dim_override != detected_lens_out:
        print(f"  [warn] lens_out_dim={lens_out_dim_override} overrides disk-detected {detected_lens_out}.")
        lens_out = lens_out_dim_override
    else:
        lens_out = detected_lens_out

    use_vit_head = (lens_out == EMBED_DIM)  # 1024 -> head; 1000 -> direct logits

    lens_model = MultiLayerLens(
        num_layers=NUM_LAYERS, dim=EMBED_DIM, mapper_type=mapper_type,
        per_layer_ranks=per_layer_ranks, lens_out_dim=lens_out,
        direct_lens_layer=(direct_lens_layer if mapper_type == "direct_lens" else None),
        lens_mlp_hidden=(lens_mlp_hidden if mapper_type in ("full", "low_rank") else None),
    ).to(device)

    load_per_layer_weights(
        lens_model, mapper_dir=mapper_dir, lens_dir=lens_dir, num_layers=NUM_LAYERS,
        mapper_type=mapper_type, device=device, unified_dir=unified_dir,
        direct_lens_path=direct_lens_path, direct_lens_layer=direct_lens_layer,
    )
    lens_model.eval()

    logits_fn = make_logits_fn(clip_model, mapper_type, use_vit_head)
    return lens_model, logits_fn, effective_layers
