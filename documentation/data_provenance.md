# Data Provenance: DeepLift Attribution Analysis

## Chain of Custody

```
data/formatted_real_PPIs.csv   (1,084 real Y2H pairs)
data/formatted_random_PPIs.csv (1,084 random MED4 pairs)
          │
          ▼
analysis/deeplift/captum_deeplift_proper_analysis.py
  Method: Captum DeepLift (reference-based backpropagation)
  Model:  ppiGPT (GPT-2, 12L/12H/768D, ~85M params; created by K. Salehi-Ashtiani)
  Ckpt:   model/out_3e/ckpt.pt (hosted on HuggingFace)
  Date:   2025-06-29
          │
          ▼
results/deeplift_motif_analysis_results.pkl  (78 MB, hosted on HuggingFace)
  Contains: per-residue attributions, interaction probabilities,
            motif discovery, position-wise statistics for all 2,168 pairs
          │
          ├──▶ visualization/create_explainability_dashboard.py
          │      Reads pkl → computes t-test, Cohen's d, Mann-Whitney U,
          │      KS test, correlations, motif counts
          │      Output: figures/DeepLift-gptPPI-sixPanel.pdf
          │
          └──▶ visualization/create_pair_attribution_heatmap.py
                 Reads pkl → selects 50 diverse pairs per dataset
                 (evenly spaced rank positions from prediction-sorted results)
                 Output: figures/DeepLift-gptPPI-FIGURE.pdf
```

## Data Integrity Certification

- No synthetic, simulated, or randomly generated data was used as a substitute
  for real experimental results in any analysis or visualization.
- All attribution values derive from Captum DeepLift applied to the trained
  ppiGPT model checkpoint on real protein sequence inputs.
- Random protein pairs (RRS) are genuine random pairings of MED4 proteins,
  not simulated interaction data.
