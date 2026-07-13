"""Attention-sink control: does the ~4-8 "gate" the lens found reflect genuine
specificity-coding content, or is it an artifact of position-0 being an
attention sink?

Uses the MED4_100 PRS/RRS sets and their seed42 "+50" counterparts (50 random
amino acids prepended to *both* proteins in every pair; same fitted lens, no
refitting) to ask: when the true N-terminus is pushed from absolute position
0-14 to ~50-64, does the representational "interacts" signal move with it
(content-tracking -> genuine specificity coding) or stay pinned at the new
absolute position 0-1 regardless of what's actually there (position-tracking
-> attention-sink artifact)?

Three readouts, all at layer 9 (the layer identified in the earlier analysis):
  A. Baseline fine-grained: PRS vs RRS, protein1 relative positions 0-14.
  B. +50 fine-grained: PRS+50 vs RRS+50, protein1 relative positions 0-69
     (continuous -- covers both the new prefix window 0-14 and the shifted
     true N-terminus ~50-64 in one sweep, so the peak's location is directly
     visible rather than only sampled at two disjoint windows).
  C. Coarse region x layer, RRS alone and RRS+50 alone (not a real-vs-random
     differential -- this is about how the negative "does not interact" call
     is itself represented, and whether the +50 prefix disturbs it): prefix /
     n_term / middle / c_term windows, layers 0-10.
"""
import pathlib
import sys

import pandas as pd
import torch
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor_jacobian_lens"))
sys.path.insert(0, str(HERE))
from ppi_lens_adapter import PPIGPLMLensModel  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

REPO = str(REPO_ROOT)
OUT_DIR = str(REPO_ROOT / "results")
DATA_DIR = str(REPO_ROOT / "data" / "nterm_prefix_control")
MARKER1_LEN = len("<ps1>,")
MARKER2_LEN = len(",<ps2>,")
LAYER = 9
N_WINDOW = 15
PREFIX_LEN = 50
MAX_SEQ_LEN = 2200


def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"{DATA_DIR}/{name}", header=None)
    df.columns = ["m1", "p1", "m2", "p2", "end"]
    assert (df.m1 == "<ps1>").all() and (df.m2 == "<ps2>").all() and (df.end == "<").all()
    df["prompt"] = "<ps1>," + df.p1 + ",<ps2>," + df.p2 + ",<"
    return df


# ---------------------------------------------------------------- A/B: fine-grained
def finegrained_baseline(model, lens, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        n = len(row.p1)
        w = min(N_WINDOW, n)
        positions = list(range(MARKER1_LEN, MARKER1_LEN + w))
        if max(positions) >= min(len(row.prompt), MAX_SEQ_LEN):
            continue
        lens_logits, _, _ = lens.apply(
            model, row.prompt, layers=[LAYER], positions=positions, max_seq_len=MAX_SEQ_LEN
        )
        for rel_pos, abs_pos in enumerate(positions):
            v = lens_logits[LAYER][rel_pos]
            margin = (v[model.token_1_id] - v[model.token_0_id]).item()
            rows.append({"dataset": dataset_name, "pair_idx": i, "rel_position": rel_pos, "margin": margin})
    return pd.DataFrame(rows)


def finegrained_plus50(model, lens, df: pd.DataFrame, dataset_name: str, max_rel: int = 70) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        n = len(row.p1)  # includes the +50 prefix
        w = min(max_rel, n)
        positions = list(range(MARKER1_LEN, MARKER1_LEN + w))
        if max(positions) >= min(len(row.prompt), MAX_SEQ_LEN):
            continue
        lens_logits, _, _ = lens.apply(
            model, row.prompt, layers=[LAYER], positions=positions, max_seq_len=MAX_SEQ_LEN
        )
        for rel_pos, abs_pos in enumerate(positions):
            v = lens_logits[LAYER][rel_pos]
            margin = (v[model.token_1_id] - v[model.token_0_id]).item()
            rows.append({"dataset": dataset_name, "pair_idx": i, "rel_position": rel_pos, "margin": margin})
        if (i + 1) % 25 == 0:
            print(f"  {dataset_name}: {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C: coarse region x layer
def region_sets_baseline(p1_len: int) -> dict[str, list[int]]:
    w = min(N_WINDOW, p1_len)
    nterm = list(range(MARKER1_LEN, MARKER1_LEN + w))
    cterm = list(range(MARKER1_LEN + p1_len - w, MARKER1_LEN + p1_len))
    mid_start = MARKER1_LEN + p1_len // 2 - w // 2
    middle = list(range(mid_start, mid_start + w)) if p1_len >= 3 * w else []
    return {"n_term": nterm, "middle": middle, "c_term": cterm}


def region_sets_plus50(p1_len_original: int, prefix_len: int = PREFIX_LEN) -> dict[str, list[int]]:
    w = min(N_WINDOW, p1_len_original)
    prefix = list(range(MARKER1_LEN, MARKER1_LEN + w))
    nterm = list(range(MARKER1_LEN + prefix_len, MARKER1_LEN + prefix_len + w))
    cterm = list(range(MARKER1_LEN + prefix_len + p1_len_original - w, MARKER1_LEN + prefix_len + p1_len_original))
    mid_start = MARKER1_LEN + prefix_len + p1_len_original // 2 - w // 2
    middle = list(range(mid_start, mid_start + w)) if p1_len_original >= 3 * w else []
    return {"prefix": prefix, "n_term": nterm, "middle": middle, "c_term": cterm}


def coarse_readout(model, lens, df: pd.DataFrame, dataset_name: str, layers, plus50: bool) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        p1_len_original = len(row.p1) - PREFIX_LEN if plus50 else len(row.p1)
        region_sets = region_sets_plus50(p1_len_original) if plus50 else region_sets_baseline(p1_len_original)
        all_positions = sorted(set(sum(region_sets.values(), [])))
        if not all_positions or max(all_positions) >= min(len(row.prompt), MAX_SEQ_LEN):
            continue
        lens_logits, _, _ = lens.apply(
            model, row.prompt, layers=layers, positions=all_positions, max_seq_len=MAX_SEQ_LEN
        )
        pos_to_idx = {p: idx for idx, p in enumerate(all_positions)}
        for region, positions in region_sets.items():
            if not positions:
                continue
            for layer in layers:
                margins = [
                    (lens_logits[layer][pos_to_idx[p]][model.token_1_id]
                     - lens_logits[layer][pos_to_idx[p]][model.token_0_id]).item()
                    for p in positions
                ]
                rows.append({
                    "dataset": dataset_name, "pair_idx": i, "region": region,
                    "layer": layer, "mean_margin": sum(margins) / len(margins),
                })
        if (i + 1) % 25 == 0:
            print(f"  coarse {dataset_name}: {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    model = PPIGPLMLensModel(f"{REPO}/model/out_3e/ckpt.pt", f"{REPO}/model/data/meta.pkl", device="cuda")
    lens = JacobianLens.load(f"{OUT_DIR}/ppiGPLM_jacobian_lens.pt")
    layers = lens.source_layers
    print("lens:", lens, "layers:", layers, flush=True)

    prs = load_csv("MED4_100_PRS.csv")
    rrs = load_csv("MED4_100_RRS.csv")
    prs50 = load_csv("MED4_100_PRS_seed42_n-term.csv")
    rrs50 = load_csv("MED4_100_RRS_seed42_n-term.csv")

    # sanity: model's own P(interact), all four sets
    with torch.no_grad():
        for name, df in [("PRS", prs), ("RRS", rrs), ("PRS+50", prs50), ("RRS+50", rrs50)]:
            probs = []
            for _, row in df.iterrows():
                ids = model.encode(row.prompt, max_length=MAX_SEQ_LEN)
                logits, _ = model.model(ids)
                p = torch.softmax(logits[0, -1], dim=0)[model.token_1_id].item()
                probs.append(p)
            print(f"{name}: mean P(interact) = {sum(probs)/len(probs):.4f}  n={len(probs)}", flush=True)

    print("\n=== A: baseline fine-grained ===", flush=True)
    df_a_prs = finegrained_baseline(model, lens, prs, "PRS")
    df_a_rrs = finegrained_baseline(model, lens, rrs, "RRS")
    df_a = pd.concat([df_a_prs, df_a_rrs], ignore_index=True)
    df_a.to_csv(f"{OUT_DIR}/ppiGPLM_prefix_control_baseline_finegrained.csv", index=False)
    print(f"saved {len(df_a)} rows", flush=True)

    print("\n=== B: +50 fine-grained (continuous 0-69) ===", flush=True)
    df_b_prs = finegrained_plus50(model, lens, prs50, "PRS+50")
    df_b_rrs = finegrained_plus50(model, lens, rrs50, "RRS+50")
    df_b = pd.concat([df_b_prs, df_b_rrs], ignore_index=True)
    df_b.to_csv(f"{OUT_DIR}/ppiGPLM_prefix_control_plus50_finegrained.csv", index=False)
    print(f"saved {len(df_b)} rows", flush=True)

    print("\n=== C: coarse region x layer, RRS and RRS+50 ===", flush=True)
    df_c_rrs = coarse_readout(model, lens, rrs, "RRS", layers, plus50=False)
    df_c_rrs50 = coarse_readout(model, lens, rrs50, "RRS+50", layers, plus50=True)
    df_c = pd.concat([df_c_rrs, df_c_rrs50], ignore_index=True)
    df_c.to_csv(f"{OUT_DIR}/ppiGPLM_prefix_control_rrs_coarse.csv", index=False)
    print(f"saved {len(df_c)} rows", flush=True)

    # ---- summary printouts ----
    print("\n=== A summary: baseline PRS vs RRS, rel 0-14, layer 9 ===")
    for pos in range(15):
        sub = df_a[df_a.rel_position == pos]
        real = sub[sub.dataset == "PRS"]["margin"]
        rand = sub[sub.dataset == "RRS"]["margin"]
        if len(real) < 3 or len(rand) < 3:
            continue
        u, p = stats.mannwhitneyu(real, rand, alternative="two-sided")
        d = (real.mean() - rand.mean()) / ((real.std() ** 2 + rand.std() ** 2) / 2) ** 0.5
        print(f"  pos {pos:2d}: real={real.mean():+.3f} random={rand.mean():+.3f} p={p:.3g} d={d:+.2f} n={len(real)}/{len(rand)}")

    print("\n=== B summary: +50 PRS+50 vs RRS+50, rel 0-69, layer 9 ===")
    for pos in range(70):
        sub = df_b[df_b.rel_position == pos]
        real = sub[sub.dataset == "PRS+50"]["margin"]
        rand = sub[sub.dataset == "RRS+50"]["margin"]
        if len(real) < 3 or len(rand) < 3:
            continue
        u, p = stats.mannwhitneyu(real, rand, alternative="two-sided")
        d = (real.mean() - rand.mean()) / ((real.std() ** 2 + rand.std() ** 2) / 2) ** 0.5
        flag = "  <-- baseline 4-8 analog" if 54 <= pos <= 58 else ("  <-- new prefix 0-1" if pos <= 1 else "")
        print(f"  pos {pos:2d}: real={real.mean():+.3f} random={rand.mean():+.3f} p={p:.3g} d={d:+.2f} n={len(real)}/{len(rand)}{flag}")

    print("\n=== C summary: RRS vs RRS+50 coarse region margins (layer 9) ===")
    for region in ["prefix", "n_term", "middle", "c_term"]:
        for dataset in ["RRS", "RRS+50"]:
            sub = df_c[(df_c.dataset == dataset) & (df_c.region == region) & (df_c.layer == LAYER)]
            if len(sub) == 0:
                continue
            print(f"  {dataset:8s} {region:8s} L{LAYER}: mean_margin={sub.mean_margin.mean():+.3f}  n={len(sub)}")


if __name__ == "__main__":
    main()
