"""Symmetric counterpart to apply_lens_finegrained.py: per-residue lens
margin across protein2's first 15 positions at layer 9 (the peak layer
identified for protein1 and consistent with protein2's coarse pass), real
vs random pairs.
"""
import pathlib
import random
import sys

import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor_jacobian_lens"))
sys.path.insert(0, str(HERE))
from ppi_lens_adapter import PPIGPLMLensModel  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

REPO = str(REPO_ROOT)
OUT_DIR = str(REPO_ROOT / "results")
MARKER1_LEN = len("<ps1>,")
MARKER2_LEN = len(",<ps2>,")
N_WINDOW = 15
MAX_SEQ_LEN = 2048
N_SAMPLE = 150
SEED = 0
LAYER = 9


def build_pairs(csv_path):
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        prompt = f"<ps1>,{p1},<ps2>,{p2},<"
        if len(prompt) <= MAX_SEQ_LEN and len(p2) >= N_WINDOW:
            out.append((p1, p2, prompt))
    return out


def analyze(model, lens, pairs, dataset_name, n_sample):
    random.seed(SEED)
    sample = random.sample(pairs, min(n_sample, len(pairs)))
    rows = []
    for i, (p1, p2, prompt) in enumerate(sample):
        p2_start = MARKER1_LEN + len(p1) + MARKER2_LEN
        w = min(N_WINDOW, len(p2))
        positions = list(range(p2_start, p2_start + w))
        if max(positions) >= min(len(prompt), MAX_SEQ_LEN):
            continue
        lens_logits, _, _ = lens.apply(
            model, prompt, layers=[LAYER], positions=positions, max_seq_len=MAX_SEQ_LEN
        )
        for rel_pos, abs_pos in enumerate(positions):
            logit_vec = lens_logits[LAYER][rel_pos]
            margin = (logit_vec[model.token_1_id] - logit_vec[model.token_0_id]).item()
            rows.append({"dataset": dataset_name, "pair_idx": i, "rel_position": rel_pos, "margin": margin})
        if (i + 1) % 50 == 0:
            print(f"  {dataset_name}: {i+1}/{len(sample)}", flush=True)
    return pd.DataFrame(rows)


def main():
    model = PPIGPLMLensModel(f"{REPO}/model/out_3e/ckpt.pt", f"{REPO}/model/data/meta.pkl", device="cuda")
    lens = JacobianLens.load(f"{OUT_DIR}/ppiGPLM_jacobian_lens.pt")

    real_pairs = build_pairs(f"{REPO}/data/formatted_real_PPIs.csv")
    random_pairs = build_pairs(f"{REPO}/data/formatted_random_PPIs.csv")

    df_real = analyze(model, lens, real_pairs, "real_PRS", N_SAMPLE)
    df_random = analyze(model, lens, random_pairs, "random_RRS", N_SAMPLE)
    df = pd.concat([df_real, df_random], ignore_index=True)
    df.to_csv(f"{OUT_DIR}/ppiGPLM_lens_nterm_positional_protein2.csv", index=False)
    print(f"saved {len(df)} rows")

    piv = df.groupby(["dataset", "rel_position"])["margin"].mean().unstack("dataset")
    piv["real_minus_random"] = piv["real_PRS"] - piv["random_RRS"]
    print(piv.round(3))


if __name__ == "__main__":
    main()
