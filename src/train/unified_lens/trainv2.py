"""
unified_lens.trainv2
===================
Unified lens v2 -- the faithfulness-aware trainer.

Same end product as v1 (one affine per ViT block), but in addition to the v1 proxy
(kept as a stabilizer) it adds a DIFFERENTIABLE deletion/insertion surrogate that
reproduces the faithfulness eval operation exactly and backprops it into (W, b):

    per-patch importance a_i  ->  soft keep-mask m_i in [0,1]
    ->  masked image  x' = m*x + (1-m)*baseline   (deletion->0, insertion->blur)
    ->  frozen ViT(x')  ->  p(c)

The ViT stays frozen; gradients still flow through it back to a_i, hence to (W, b).
A light concentration penalty sharpens FG importance (helps deletion); a proxy-only
warmup gives the surrogate a sane starting map; stochastic fraction sampling gives an
unbiased AUC estimate per step.

Run:
    python -m train.unified_lens.trainv2 --config configs/train/unified_lens/v2.yaml
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import UnifiedLensConfig
from .data import FGBGMaskDataset, resolve_data_dir
from .lens import AffineLens, MultiLayerHook, build_head_fn, load_vit, make_blurred_baseline


def proxy_patch_loss(student_logits, teacher_logits, mask, T=3.0, eps=1e-8):
    """v1 proxy, kept as a stabilizing regularizer. Returns (fg_loss, bg_loss)."""
    B, N, C = student_logits.shape
    s_log_T = F.log_softmax(student_logits / T, dim=-1)
    t_log_T = F.log_softmax(teacher_logits / T, dim=-1).unsqueeze(1)
    t_p_T = t_log_T.exp()
    kl = (t_p_T * (t_log_T - s_log_T)).sum(dim=-1) * (T * T)                # (B,N)

    s_log_1 = F.log_softmax(student_logits, dim=-1)
    H = -(s_log_1.exp() * s_log_1).sum(dim=-1)                              # (B,N)
    bg_pen = math.log(C) - H

    fg = (mask > 0.5).float()
    bg = 1.0 - fg
    fg_loss = (kl * fg).sum() / (fg.sum() + eps)
    bg_loss = (bg_pen * bg).sum() / (bg.sum() + eps)
    return fg_loss, bg_loss


def concentration_loss(prob_c, mask, eps=1e-8):
    """Normalized entropy of the FG importance distribution (lower = peakier)."""
    fg = (mask > 0.5).float()
    n_fg = fg.sum(dim=1)
    valid = (n_fg >= 2.0).float()
    w = prob_c * fg
    Z = w.sum(dim=1, keepdim=True) + eps
    p = w / Z
    H = -(p * torch.log(p + eps) * fg).sum(dim=1)
    denom = torch.log(n_fg.clamp(min=2.0))
    norm_H = (H / denom) * valid
    return norm_H.sum() / valid.sum().clamp(min=1.0)


def soft_keep_mask(importance, frac, mode, beta, eps=1e-6):
    """Smooth per-patch keep-mask (B, N). mode: 'insertion' keep top frac; 'deletion' remove top."""
    mu = importance.mean(dim=1, keepdim=True)
    sd = importance.std(dim=1, keepdim=True) + eps
    z = (importance - mu) / sd
    q = torch.quantile(z.detach(), 1.0 - frac, dim=1, keepdim=True)
    top = torch.sigmoid(beta * (z - q))
    return top if mode == "insertion" else (1.0 - top)


def upsample_patch_mask(keep, grid, patch):
    """(B, N) patch keep-mask -> (B, 1, H, W) pixel keep-mask (nearest)."""
    B = keep.shape[0]
    m = keep.view(B, 1, grid, grid)
    return F.interpolate(m, scale_factor=patch, mode="nearest")


def train(cfg: UnifiedLensConfig, vit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------- data ----------
    data_dir = resolve_data_dir(cfg.mask_source, cfg.data_dir)
    full = FGBGMaskDataset(data_dir, img_size=224, patch_size=14)
    grid = full.grid
    patch = 224 // grid
    print(f"[data] mask_source={cfg.mask_source} dir={data_dir} "
          f"{len(full)} pairs, grid={grid}x{grid}, patch={patch}")

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
    if cfg.grad_checkpoint and hasattr(vit, "set_grad_checkpointing"):
        vit.set_grad_checkpointing(True)
    head_fn = build_head_fn(vit)
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

    # Teacher forward in fp32; the memory-heavy masked surrogate forward uses bf16 if
    # available (same exponent range as fp32, no overflow) and fp32 otherwise.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def surrogate_autocast():
        if device.type == "cuda" and use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    print(f"[amp] teacher forward=fp32; surrogate forward={'bf16' if use_bf16 else 'fp32'}")

    def faithfulness_loss(imgs, baseline_blur, importance, target_idx, training):
        """Eval-matched differentiable deletion/insertion surrogate (single batched forward)."""
        B = imgs.shape[0]
        fr = list(cfg.faith_fractions)
        if cfg.faith_samples < len(fr):
            sel = torch.randperm(len(fr))[:cfg.faith_samples].tolist()
            fr = [fr[i] for i in sel]

        zeros = torch.zeros_like(imgs)
        masked, tags = [], []   # +1 deletion, -1 insertion
        for f in fr:
            keep_del = soft_keep_mask(importance, f, "deletion", cfg.faith_beta)
            keep_ins = soft_keep_mask(importance, f, "insertion", cfg.faith_beta)
            md = upsample_patch_mask(keep_del, grid, patch)
            mi = upsample_patch_mask(keep_ins, grid, patch)
            masked.append(md * imgs + (1.0 - md) * zeros)          # deletion: ->0
            masked.append(mi * imgs + (1.0 - mi) * baseline_blur)  # insertion: ->blur
            tags += [1, -1]
        big = torch.cat(masked, dim=0)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx, surrogate_autocast():
            logits = vit(big)
        logits = logits.float()
        p = F.softmax(logits, dim=-1)
        tgt = target_idx.repeat(len(masked))
        ar = torch.arange(p.shape[0], device=device)
        p_c = p[ar, tgt].view(len(masked), B)

        tags_t = torch.tensor(tags, device=device, dtype=p_c.dtype)
        del_term = p_c[tags_t > 0].mean()
        ins_term = p_c[tags_t < 0].mean()
        loss = cfg.del_weight * del_term + cfg.ins_weight * (1.0 - ins_term)
        return loss, del_term.detach(), ins_term.detach()

    def run_step(imgs, masks, training, use_faith):
        with torch.no_grad():
            teacher_logits = vit(imgs).float()
        target_idx = teacher_logits.argmax(dim=-1)               # == y_hat baseline
        patch_tok = {li: hooks.cache[li].float() for li in layers}
        baseline_blur = make_blurred_baseline(imgs) if use_faith and cfg.faith_weight > 0 else None

        out = {}
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        for li in layers:
            if stopped[li]:
                continue
            with grad_ctx:
                h = patch_tok[li]
                student_logits = head_fn(lenses[li](h))          # (B,N,C)
                logp = F.log_softmax(student_logits, dim=-1)
                prob = logp.exp()
                ar = torch.arange(h.shape[0], device=device)
                importance = logp[ar][:, :, :].gather(
                    2, target_idx.view(-1, 1, 1).expand(-1, h.shape[1], 1)).squeeze(-1)
                prob_c = prob[ar][:, :, :].gather(
                    2, target_idx.view(-1, 1, 1).expand(-1, h.shape[1], 1)).squeeze(-1)

                fg_loss, bg_loss = proxy_patch_loss(
                    student_logits, teacher_logits, masks, T=cfg.temperature)
                conc = concentration_loss(prob_c, masks)
                proxy = fg_loss + cfg.bg_weight * bg_loss + cfg.conc_weight * conc

                if use_faith and cfg.faith_weight > 0:
                    f_loss, del_t, ins_t = faithfulness_loss(
                        imgs, baseline_blur, importance, target_idx, training)
                else:
                    f_loss = torch.zeros((), device=device)
                    del_t = torch.zeros(()); ins_t = torch.zeros(())

                loss = proxy + cfg.faith_weight * f_loss

            if training:
                opts[li].zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lenses[li].parameters(), cfg.grad_clip)
                opts[li].step()

            out[li] = (loss.item(), fg_loss.item(), bg_loss.item(),
                       conc.item(), float(del_t), float(ins_t))
        return out

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        use_faith = epoch > cfg.warmup_epochs
        phase = "faith" if use_faith else "warmup"

        for li in layers:
            lenses[li].train()
        tr = {li: [0.0] * 6 + [0] for li in layers}
        pbar = tqdm(train_loader, desc=f"ep {epoch:02d} {phase} train", ncols=120, leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            res = run_step(imgs, masks, training=True, use_faith=use_faith)
            for li, vals in res.items():
                for k in range(6):
                    tr[li][k] += vals[k]
                tr[li][6] += 1
            shown = next((li for li in layers if not stopped[li]), None)
            if shown is not None and tr[shown][6] > 0:
                s = tr[shown]; n = s[6]
                pbar.set_postfix(L=f"{s[0]/n:.3f}", fg=f"{s[1]/n:.3f}", bg=f"{s[2]/n:.3f}",
                                 cc=f"{s[3]/n:.3f}", d=f"{s[4]/n:.3f}", i=f"{s[5]/n:.3f}", ly=shown)

        for li in layers:
            lenses[li].eval()
        va = {li: [0.0] * 6 + [0] for li in layers}
        for imgs, masks in tqdm(val_loader, desc=f"ep {epoch:02d} {phase} val  ", ncols=120, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            res = run_step(imgs, masks, training=False, use_faith=use_faith)
            for li, vals in res.items():
                for k in range(6):
                    va[li][k] += vals[k]
                va[li][6] += 1

        epoch_log = {"epoch": epoch, "phase": phase,
                     "elapsed_s": round(time.time() - t0, 1), "layers": {}}
        msg = [f"\n[epoch {epoch:02d} | {phase}]  elapsed={epoch_log['elapsed_s']}s"]
        for li in layers:
            nt = max(tr[li][6], 1); nv = max(va[li][6], 1)
            tl = tr[li][0] / nt; vl = va[li][0] / nv
            v_fg, v_bg, v_cc, v_d, v_i = (va[li][k] / nv for k in range(1, 6))
            epoch_log["layers"][li] = dict(
                train_loss=tl, val_loss=vl, val_fg=v_fg, val_bg=v_bg,
                val_conc=v_cc, val_del=v_d, val_ins=v_i,
                stopped=stopped[li], best_val=best_val[li])
            tag = " (stopped)" if stopped[li] else ""
            line = (f"  L{li:02d}{tag}  train {tl:.4f}  val {vl:.4f} "
                    f"(fg {v_fg:.4f} / bg {v_bg:.4f} / cc {v_cc:.4f} / "
                    f"del {v_d:.4f} / ins {v_i:.4f})")
            if not stopped[li]:
                if vl < best_val[li] - 1e-5:
                    best_val[li] = vl
                    bad_epochs[li] = 0
                    torch.save({
                        "epoch": epoch, "layer_idx": li, "val_loss": vl,
                        "val_fg": v_fg, "val_bg": v_bg, "val_conc": v_cc,
                        "val_del": v_d, "val_ins": v_i,
                        "config": cfg.to_dict(), "state_dict": lenses[li].state_dict(),
                        "model_name": cfg.model_name, "dim": dim,
                    }, save_root / f"layer_{li:02d}_best.pt")
                    line += f"  -> SAVED (best {best_val[li]:.4f})"
                else:
                    if use_faith or cfg.warmup_epochs == 0:
                        bad_epochs[li] += 1
                        if bad_epochs[li] >= cfg.patience:
                            stopped[li] = True
                            line += f"  -> early stop (best {best_val[li]:.4f})"
            msg.append(line)
        print("\n".join(msg))

        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        if epoch == cfg.warmup_epochs and cfg.faith_weight > 0:
            best_val = {li: float("inf") for li in layers}
            bad_epochs = {li: 0 for li in layers}
            print("[curriculum] warmup done -> enabling faithfulness surrogate; "
                  "resetting best_val / early-stop counters.")

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
    p = argparse.ArgumentParser(description="Unified lens v2 (faithfulness-aware) trainer.")
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
