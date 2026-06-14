"""
unified_lens.trainv1
===================
Unified lens v1 -- the proxy-only trainer.

Trains a single ``dim x dim`` affine (Wx + b) per ViT block that maps a *patch* token
into classifier-ready space. Per patch, averaged separately over FG / BG then summed:

    FG patches:  T^2 * KL( softmax(y_hat / T) || softmax(student / T) )   (distill to y_hat)
    BG patches:  log(C) - H(student)                                       (push to uniform)
    total = fg_loss + bg_weight * bg_loss

FG/BG come from the ``mask_source`` masks (ImageNet-S segmentation or ImageNet box union).
The backbone is frozen; only the per-layer affine is trained.

Run:
    python -m train.unified_lens.trainv1 --config configs/train/unified_lens/v1.yaml
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import UnifiedLensConfig
from .data import FGBGMaskDataset, resolve_data_dir
from .lens import AffineLens, MultiLayerHook, build_head_fn, load_vit


def dual_patch_loss(student_logits, teacher_logits, mask, T=3.0, bg_weight=1.0, eps=1e-8):
    """FG: temperature-scaled KL distillation toward y_hat. BG: push toward uniform.

    student_logits : (B, N, C)   teacher_logits : (B, C)   mask : (B, N) FG=1
    """
    B, N, C = student_logits.shape

    # Numerically safe KL(teacher || student) at temperature T (log-space, no NaN).
    s_log_T = F.log_softmax(student_logits / T, dim=-1)
    t_log_T = F.log_softmax(teacher_logits / T, dim=-1).unsqueeze(1)
    t_p_T = t_log_T.exp()
    kl = (t_p_T * (t_log_T - s_log_T)).sum(dim=-1) * (T * T)                # (B, N)

    s_log_1 = F.log_softmax(student_logits, dim=-1)
    H = -(s_log_1.exp() * s_log_1).sum(dim=-1)                              # (B, N)
    bg_pen = math.log(C) - H                                                # >= 0, 0 iff uniform

    fg = (mask > 0.5).float()
    bg = 1.0 - fg
    fg_loss = (kl * fg).sum() / (fg.sum() + eps)
    bg_loss = (bg_pen * bg).sum() / (bg.sum() + eps)

    total = fg_loss + bg_weight * bg_loss
    return total, fg_loss.detach(), bg_loss.detach()


def train(cfg: UnifiedLensConfig, vit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------- data ----------
    data_dir = resolve_data_dir(cfg.mask_source, cfg.data_dir)
    full = FGBGMaskDataset(data_dir, img_size=224, patch_size=14)
    print(f"[data] mask_source={cfg.mask_source} dir={data_dir} "
          f"{len(full)} pairs, grid={full.grid}x{full.grid}")

    val_n = max(1, int(cfg.val_frac * len(full)))
    train_n = len(full) - val_n
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full, [train_n, val_n], generator=g)
    print(f"[data] train={train_n}  val={val_n}")

    dl_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.workers,
                     pin_memory=True, persistent_workers=cfg.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # ---------- model ----------
    if vit is None:
        print(f"[model] loading {cfg.model_name} ...")
        vit = load_vit(cfg.model_name, cfg.model_weights_path, device)
    head_fn = build_head_fn(vit)                       # norm -> fc_norm -> head
    dim = vit.head.in_features
    n_classes = vit.head.out_features
    n_blocks = len(vit.blocks)
    print(f"[model] dim={dim}, classes={n_classes}, blocks={n_blocks}")

    layers = sorted(set(int(l) for l in cfg.layers))
    bad = [l for l in layers if not (0 <= l < n_blocks)]
    if bad:
        raise ValueError(f"Layers {bad} out of range [0, {n_blocks - 1}]")
    print(f"[layers] training affines for blocks: {layers}")

    lenses = {li: AffineLens(dim=dim, init_identity=True).to(device) for li in layers}
    opts = {li: torch.optim.AdamW(lenses[li].parameters(), lr=cfg.lr, weight_decay=cfg.wd)
            for li in layers}
    hooks = MultiLayerHook(vit, layers)

    best_val = {li: float("inf") for li in layers}
    bad_epochs = {li: 0 for li in layers}
    stopped = {li: False for li in layers}

    save_root = Path(cfg.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    log_path = save_root / "train_log.jsonl"
    with open(save_root / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[io] saving to {save_root}")

    def run_layers_for_loss(patch_tok_dict, teacher_logits, masks, training):
        out = {}
        for li in layers:
            if stopped[li]:
                continue
            lens = lenses[li]
            student_logits = head_fn(lens(patch_tok_dict[li]))     # (B, N, C)
            loss, fg_l, bg_l = dual_patch_loss(
                student_logits, teacher_logits, masks,
                T=cfg.temperature, bg_weight=cfg.bg_weight,
            )
            if training:
                opts[li].zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lens.parameters(), cfg.grad_clip)
                opts[li].step()
            out[li] = (loss.item(), fg_l.item(), bg_l.item())
        return out

    def run_epoch(loader, desc, training):
        for li in layers:
            lenses[li].train(training)
        stats = {li: [0.0, 0.0, 0.0, 0] for li in layers}
        pbar = tqdm(loader, desc=desc, ncols=120, leave=False)
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            # Teacher forward in fp32 (the CLIP ViT-L overflows in fp16).
            with torch.no_grad():
                teacher_logits = vit(imgs).float()
            patch_tok = {li: hooks.cache[li].float() for li in layers}
            with grad_ctx:
                results = run_layers_for_loss(patch_tok, teacher_logits, masks, training)
            for li, (lo, fg, bg) in results.items():
                s = stats[li]
                s[0] += lo; s[1] += fg; s[2] += bg; s[3] += 1
            shown = next((li for li in layers if not stopped[li]), None)
            if shown is not None and stats[shown][3] > 0:
                s = stats[shown]
                pbar.set_postfix(L=f"{s[0]/s[3]:.3f}", fg=f"{s[1]/s[3]:.3f}",
                                 bg=f"{s[2]/s[3]:.3f}", layer=shown)
        return stats

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(train_loader, f"epoch {epoch:02d} train", True)
        val_stats = run_epoch(val_loader, f"epoch {epoch:02d} val  ", False)

        epoch_log = {"epoch": epoch, "elapsed_s": round(time.time() - t0, 1), "layers": {}}
        msg = [f"\n[epoch {epoch:02d}]  elapsed={epoch_log['elapsed_s']}s"]
        for li in layers:
            n_tr = max(train_stats[li][3], 1)
            n_va = max(val_stats[li][3], 1)
            tr_loss, tr_fg, tr_bg = (train_stats[li][k] / n_tr for k in range(3))
            va_loss, va_fg, va_bg = (val_stats[li][k] / n_va for k in range(3))
            epoch_log["layers"][li] = dict(
                train_loss=tr_loss, train_fg=tr_fg, train_bg=tr_bg,
                val_loss=va_loss, val_fg=va_fg, val_bg=va_bg,
                stopped=stopped[li], best_val=best_val[li])
            tag = " (stopped)" if stopped[li] else ""
            line = (f"  L{li:02d}{tag}  train {tr_loss:.4f} (fg {tr_fg:.4f} / bg {tr_bg:.4f})"
                    f"  val {va_loss:.4f} (fg {va_fg:.4f} / bg {va_bg:.4f})")
            if not stopped[li]:
                if va_loss < best_val[li] - 1e-5:
                    best_val[li] = va_loss
                    bad_epochs[li] = 0
                    torch.save({
                        "epoch": epoch, "layer_idx": li, "val_loss": va_loss,
                        "val_fg": va_fg, "val_bg": va_bg, "config": cfg.to_dict(),
                        "state_dict": lenses[li].state_dict(),
                        "model_name": cfg.model_name, "dim": dim,
                    }, save_root / f"layer_{li:02d}_best.pt")
                    line += f"  -> SAVED (best {best_val[li]:.4f})"
                else:
                    bad_epochs[li] += 1
                    if bad_epochs[li] >= cfg.patience:
                        stopped[li] = True
                        line += f"  -> early stop (best {best_val[li]:.4f})"
            msg.append(line)
        print("\n".join(msg))

        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        if all(stopped.values()):
            print("[done] all layers early-stopped.")
            break

    hooks.remove()
    summary = {li: {"best_val": best_val[li], "stopped": stopped[li]} for li in layers}
    with open(save_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[summary]")
    for li in layers:
        print(f"  L{li:02d}  best_val={best_val[li]:.4f}  stopped={stopped[li]}")
    print(f"[done] total wall time: {(time.time() - t0)/60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Unified lens v1 (proxy-only) trainer.")
    p.add_argument("--config", required=True, help="path to a unified_lens YAML config")
    p.add_argument("--mask-source", choices=["imagenet_s", "box"], default=None,
                   help="override config mask_source")
    p.add_argument("--data-dir", default=None, help="override config data_dir")
    p.add_argument("--save-dir", default=None, help="override config save_dir")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="override config layers")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = UnifiedLensConfig.from_yaml(
        args.config, mask_source=args.mask_source, data_dir=args.data_dir,
        save_dir=args.save_dir, layers=args.layers,
    )
    train(cfg)


if __name__ == "__main__":
    main()
