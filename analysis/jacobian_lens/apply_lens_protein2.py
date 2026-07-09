"""Symmetric counterpart to apply_lens.py: same coarse region x layer lens
readout (N-terminal / middle / C-terminal 15-residue windows, real vs random
pairs), but for protein2 instead of protein1.

Motivation: protein1's N-terminus showed a representation-level "interacts"
signal that was absent at its very first 1-2 residues (position 0-1) despite
those residues carrying the highest DeepLift attribution in the manuscript.
Because ppiGPLM is a causal decoder, protein2 sits downstream of protein1 in
the prompt ("<ps1>,{protein1},<ps2>,{protein2},<") and is causally much
closer to the final decision token, so its position dynamics need not mirror
protein1's -- this script tests whether they do.
"""
import pathlib
import random
import sys

import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent  # analysis/jacobian_lens
REPO_ROOT = HERE.parents[1]  # repo root
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
N_SAMPLE = 200
SEED = 0


def build_pairs(csv_path: str) -> list[tuple[str, str, str]]:
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        prompt = f"<ps1>,{p1},<ps2>,{p2},<"
        if len(prompt) <= MAX_SEQ_LEN:
            out.append((p1, p2, prompt))
    return out


def position_sets_p2(p1: str, p2: str) -> dict[str, list[int]]:
    """Absolute prompt positions for protein2's N-term/C-term/middle windows."""
    p2_start = MARKER1_LEN + len(p1) + MARKER2_LEN
    n = len(p2)
    w = min(N_WINDOW, n)
    nterm = list(range(p2_start, p2_start + w))
    cterm = list(range(p2_start + n - w, p2_start + n))
    mid_start = p2_start + n // 2 - w // 2
    middle = list(range(mid_start, mid_start + w)) if n >= 3 * w else []
    return {"n_term": nterm, "c_term": cterm, "middle": middle}


def analyze_dataset(model, lens, pairs, dataset_name, n_sample, layers) -> pd.DataFrame:
    random.seed(SEED)
    usable = [pr for pr in pairs if len(pr[1]) >= 20]  # protein2 needs room for windows
    sample = random.sample(usable, min(n_sample, len(usable)))
    rows = []
    for i, (p1, p2, prompt) in enumerate(sample):
        pos_sets = position_sets_p2(p1, p2)
        all_positions = sorted(set(sum(pos_sets.values(), [])))
        if not all_positions or max(all_positions) >= min(len(prompt), MAX_SEQ_LEN):
            continue
        lens_logits, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, positions=all_positions, max_seq_len=MAX_SEQ_LEN
        )
        pos_to_idx = {p: idx for idx, p in enumerate(all_positions)}

        with torch.no_grad():
            h = model.forward(input_ids)
            final_logits = model.unembed(h[:, [-1], :])[0, 0]
            final_prob1 = torch.softmax(final_logits, dim=0)[model.token_1_id].item()

        for region, positions in pos_sets.items():
            if not positions:
                continue
            for layer in layers:
                margins = []
                for p in positions:
                    logit_vec = lens_logits[layer][pos_to_idx[p]]
                    margin = (logit_vec[model.token_1_id] - logit_vec[model.token_0_id]).item()
                    margins.append(margin)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "pair_idx": i,
                        "protein": "protein2",
                        "region": region,
                        "layer": layer,
                        "mean_margin": sum(margins) / len(margins),
                        "final_model_prob1": final_prob1,
                    }
                )
        if (i + 1) % 25 == 0:
            print(f"  {dataset_name}: {i+1}/{len(sample)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    model = PPIGPLMLensModel(
        f"{REPO}/model/out_3e/ckpt.pt",
        f"{REPO}/model/data/meta.pkl",
        device="cuda",
    )
    lens = JacobianLens.load(f"{OUT_DIR}/ppiGPLM_jacobian_lens.pt")
    layers = lens.source_layers
    print("lens:", lens, "layers:", layers, flush=True)

    real_pairs = build_pairs(f"{REPO}/data/formatted_real_PPIs.csv")
    random_pairs = build_pairs(f"{REPO}/data/formatted_random_PPIs.csv")
    print(f"real pairs usable: {len(real_pairs)}, random pairs usable: {len(random_pairs)}", flush=True)

    df_real = analyze_dataset(model, lens, real_pairs, "real_PRS", N_SAMPLE, layers)
    df_random = analyze_dataset(model, lens, random_pairs, "random_RRS", N_SAMPLE, layers)
    df = pd.concat([df_real, df_random], ignore_index=True)
    df.to_csv(f"{OUT_DIR}/ppiGPLM_lens_readout_protein2.csv", index=False)
    print(f"saved {len(df)} rows to ppiGPLM_lens_readout_protein2.csv", flush=True)


if __name__ == "__main__":
    main()
