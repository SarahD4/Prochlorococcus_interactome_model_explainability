"""Lens-coordinate-swapping / steering experiment on ppiGPLM.

The prior lens readouts (apply_lens.py / apply_lens_finegrained.py, and their
protein2 counterparts) are *correlational*: they ask whether a position's
residual, transported into the final-layer basis, already looks like the
"interacts" decision. This script asks the causal question directly, by
literally overwriting the model's own residual stream and watching the real
output change -- classic activation patching, operationalized as the
"steering intervention" jacobian-lens' source paper describes.

Procedure, for a donor prompt and a recipient prompt:
  1. Run both forward through block 9 (LAYER) to get their raw residual
     streams at that point -- the same layer the lens was fit/applied at.
  2. Clone the recipient's residual, and overwrite it at a chosen set of
     positions with the donor's residual at the analogous (relative-to-
     protein-start) positions.
  3. Continue the forward pass (blocks 10, 11, then unembed) on both the
     recipient's own (unpatched) residual and the patched one.
  4. shift = patched_margin - baseline_margin, where margin =
     logit('1') - logit('0') at the final prompt position.

Patching happens in the model's own raw residual basis, not the
lens-transported (unembed(J_l @ h)) space: the raw residual is what the rest
of the network actually consumes, so it's the causally correct thing to
intervene on. The lens (and its earlier readouts) is what told us *where* to
intervene -- layer 9, and which positions -- not how to perform the
intervention itself.

Two position sets per protein:
  - "peak":    the positions with the strongest lens-margin real-vs-random
               differential at layer 9 (relative 6,7,8 for both proteins).
  - "control": positions with a near-null lens-margin differential --
               relative 0,1 for protein1 (both were null); relative 1 only
               for protein2 (position 0 was weakly non-null in the earlier
               pass, attributed to cross-attention onto protein1, so it is
               excluded from the negative control here).
"""
import pathlib
import random
import sys

import pandas as pd
import torch
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor_jacobian_lens"))
sys.path.insert(0, str(HERE))
from ppi_lens_adapter import PPIGPLMLensModel  # noqa: E402

REPO = str(REPO_ROOT)
OUT_DIR = str(REPO_ROOT / "results")
MARKER1_LEN = len("<ps1>,")
MARKER2_LEN = len(",<ps2>,")
LAYER = 9  # output of transformer.h[9]; same layer the lens readouts used
N_SAMPLE = 150
SEED = 0
MAX_SEQ_LEN = 2048
MIN_PROTEIN_LEN = 20

PEAK_REL = [6, 7, 8]
CONTROL_REL_P1 = [0, 1]
CONTROL_REL_P2 = [1]


def build_pairs(csv_path: str) -> list[tuple[str, str, str]]:
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        prompt = f"<ps1>,{p1},<ps2>,{p2},<"
        if len(prompt) <= MAX_SEQ_LEN and len(p1) >= MIN_PROTEIN_LEN and len(p2) >= MIN_PROTEIN_LEN:
            out.append((p1, p2, prompt))
    return out


def encode(model, prompt: str) -> torch.Tensor:
    return model.encode(prompt, max_length=MAX_SEQ_LEN)


@torch.no_grad()
def run_to_layer(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Raw residual stream after transformer.h[layer_idx] (0-indexed)."""
    b, t = input_ids.size()
    pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
    tok_emb = model.model.transformer.wte(input_ids)
    pos_emb = model.model.transformer.wpe(pos)
    x = model.model.transformer.drop(tok_emb + pos_emb)
    for i, block in enumerate(model.model.transformer.h):
        x = block(x)
        if i == layer_idx:
            return x
    return x


@torch.no_grad()
def continue_from_layer(model, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Finish the forward pass from the output of block layer_idx; return final-position logits."""
    for i in range(layer_idx + 1, len(model.model.transformer.h)):
        x = model.model.transformer.h[i](x)
    return model.unembed(x[:, [-1], :])[0, 0]  # [vocab]


def margin(model, logits: torch.Tensor) -> float:
    return (logits[model.token_1_id] - logits[model.token_0_id]).item()


def abs_positions(rel_positions: list[int], start: int) -> list[int]:
    return [start + r for r in rel_positions]


def patch_and_measure(
    model, donor_x: torch.Tensor, recipient_x: torch.Tensor,
    donor_abs_pos: list[int], recipient_abs_pos: list[int],
) -> tuple[float, float]:
    """Returns (baseline_margin, patched_margin) for the recipient."""
    baseline_logits = continue_from_layer(model, recipient_x, LAYER)
    baseline_margin = margin(model, baseline_logits)

    patched_x = recipient_x.clone()
    for d_pos, r_pos in zip(donor_abs_pos, recipient_abs_pos, strict=True):
        patched_x[0, r_pos, :] = donor_x[0, d_pos, :]
    patched_logits = continue_from_layer(model, patched_x, LAYER)
    patched_margin = margin(model, patched_logits)
    return baseline_margin, patched_margin


def main() -> None:
    model = PPIGPLMLensModel(f"{REPO}/model/out_3e/ckpt.pt", f"{REPO}/model/data/meta.pkl", device="cuda")

    real_pairs = build_pairs(f"{REPO}/data/formatted_real_PPIs.csv")
    random_pairs = build_pairs(f"{REPO}/data/formatted_random_PPIs.csv")
    random.seed(SEED)
    real_sample = random.sample(real_pairs, min(N_SAMPLE, len(real_pairs)))
    random.seed(SEED + 1)
    random_sample = random.sample(random_pairs, min(N_SAMPLE, len(random_pairs)))
    n = min(len(real_sample), len(random_sample))
    real_sample, random_sample = real_sample[:n], random_sample[:n]
    print(f"index-matched pairs: {n}", flush=True)

    rows = []
    for i in range(n):
        r_p1, r_p2, r_prompt = real_sample[i]
        n_p1, n_p2, n_prompt = random_sample[i]  # "n" for random/noise pair

        real_ids = encode(model, r_prompt)
        rand_ids = encode(model, n_prompt)
        real_x = run_to_layer(model, real_ids, LAYER)
        rand_x = run_to_layer(model, rand_ids, LAYER)

        p1_start_real, p1_start_rand = MARKER1_LEN, MARKER1_LEN
        p2_start_real = MARKER1_LEN + len(r_p1) + MARKER2_LEN
        p2_start_rand = MARKER1_LEN + len(n_p1) + MARKER2_LEN

        conditions = [
            ("protein1", "peak", PEAK_REL, p1_start_real, p1_start_rand),
            ("protein1", "control", CONTROL_REL_P1, p1_start_real, p1_start_rand),
            ("protein2", "peak", PEAK_REL, p2_start_real, p2_start_rand),
            ("protein2", "control", CONTROL_REL_P2, p2_start_real, p2_start_rand),
        ]

        for protein, region, rel_positions, start_real, start_rand in conditions:
            real_abs = abs_positions(rel_positions, start_real)
            rand_abs = abs_positions(rel_positions, start_rand)
            if max(real_abs) >= real_ids.shape[1] or max(rand_abs) >= rand_ids.shape[1]:
                continue

            # donor=real -> recipient=random
            base_m, patched_m = patch_and_measure(model, real_x, rand_x, real_abs, rand_abs)
            rows.append({
                "pair_idx": i, "protein": protein, "region": region,
                "direction": "real_into_random", "baseline_margin": base_m,
                "patched_margin": patched_m, "shift": patched_m - base_m,
            })

            # donor=random -> recipient=real
            base_m, patched_m = patch_and_measure(model, rand_x, real_x, rand_abs, real_abs)
            rows.append({
                "pair_idx": i, "protein": protein, "region": region,
                "direction": "random_into_real", "baseline_margin": base_m,
                "patched_margin": patched_m, "shift": patched_m - base_m,
            })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/ppiGPLM_steering_swap_results.csv", index=False)
    print(f"saved {len(df)} rows to ppiGPLM_steering_swap_results.csv", flush=True)

    # ---- summary ----
    print("\n=== per-condition shift (one-sample vs 0) ===")
    summary_rows = []
    for protein in ["protein1", "protein2"]:
        for region in ["peak", "control"]:
            for direction in ["real_into_random", "random_into_real"]:
                sub = df[(df.protein == protein) & (df.region == region) & (df.direction == direction)]
                if len(sub) == 0:
                    continue
                shifts = sub["shift"]
                t, p = stats.wilcoxon(shifts)
                d = shifts.mean() / shifts.std()
                print(f"{protein:9s} {region:8s} {direction:18s} n={len(shifts):3d}  "
                      f"mean_shift={shifts.mean():+.3f}  std={shifts.std():.3f}  "
                      f"Wilcoxon p={p:.2e}  d={d:+.2f}")
                summary_rows.append({
                    "protein": protein, "region": region, "direction": direction,
                    "n": len(shifts), "mean_shift": shifts.mean(), "std_shift": shifts.std(),
                    "wilcoxon_p_vs_0": p, "cohens_d_vs_0": d,
                })

    print("\n=== peak vs control, paired comparison (|shift|) ===")
    paired_rows = []
    for protein in ["protein1", "protein2"]:
        for direction in ["real_into_random", "random_into_real"]:
            peak = df[(df.protein == protein) & (df.region == "peak") & (df.direction == direction)]
            ctrl = df[(df.protein == protein) & (df.region == "control") & (df.direction == direction)]
            merged = peak.merge(ctrl, on="pair_idx", suffixes=("_peak", "_control"))
            diff = merged["shift_peak"].abs() - merged["shift_control"].abs()
            t, p = stats.wilcoxon(diff)
            dz = diff.mean() / diff.std()
            print(f"{protein:9s} {direction:18s} n={len(diff):3d}  "
                  f"mean(|peak|-|control|)={diff.mean():+.3f}  Wilcoxon p={p:.2e}  dz={dz:+.2f}")
            paired_rows.append({
                "protein": protein, "direction": direction, "n": len(diff),
                "mean_abs_peak_minus_control": diff.mean(), "wilcoxon_p": p, "cohens_dz": dz,
            })

    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/ppiGPLM_steering_swap_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(f"{OUT_DIR}/ppiGPLM_steering_swap_paired_comparison.csv", index=False)


if __name__ == "__main__":
    main()
