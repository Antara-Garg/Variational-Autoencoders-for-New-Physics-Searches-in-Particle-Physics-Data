# Variational-Autoencoders-for-New-Physics-Searches-in-Particle-Physics-Data
Unsupervised anomaly detection for model-independent new physics searches, using the LHC Olympics 2020 R&D dataset.

## Data & Features

- **Dataset**: LHC Olympics 2020 R&D dataset (high-level jet features), a public benchmark for anomaly detection at the LHC. [arXiv:2101.08320v1](https://arxiv.org/abs/2101.08320)
- **Features**: per-jet mass, n-subjettiness ratios (τ21, τ32), transverse momentum, and other substructure observables; deliberately excluded dijet invariant mass (mjj) as a training input, since that's the variable used to define the signal region.
- **Setup**: model trained only on background-like events from a sideband region in mjj; evaluated on the signal region, where true labels are used only for evaluation.

## Methodology

1. **Baseline models**: a plain autoencoder and a variational autoencoder (VAE), both trained to reconstruct background events, using reconstruction error as the anomaly score. [arXiv:1901.03407v2](https://arxiv.org/pdf/1901.03407)
2. **Data and Feature Sets**: used feature engineering to create feature groups, and added them one by one, over a baseline set, noting down the changes in AUC, to obtain the best feature set.
3. **Mass-sculpting check**: correlating the anomaly score with jet mass, on background-only events, to test whether the model was learning genuine substructure or secretly exploiting mass as a shortcut.
4. **Finding**: certain engineered features and an oversized latent dimension caused the anomaly score to correlate strongly with jet mass (Pearson r up to ~0.48)
5. **Fix**: added a **distance correlation (DisCo) regularization** term to the VAE's loss function, directly penalizing correlation between the anomaly score and jet mass during training. [arXiv:2001.05310](https://arxiv.org/abs/2001.05310).

## Results

| Model | AUC | Correlation with mj1 (background) |
|---|---|---|
| Autoencoder (baseline) | 0.76 | — |
| VAE (uncorrected) | up to 0.80 | up to 0.48 (leaking) |
| **VAE + DisCo (corrected)** | **0.73–0.75** | **0.09–0.14** |

## Next Steps

- Derive richer substructure features (energy correlation functions, jet width) from raw particle-level data rather than high-level approximations.
- Explore hybrid models paired with autoencoders (VAE encoder + classical detectors like Isolation Forest) and one-class neural networks (OCNN).
- Extend to graph-based VAEs. [arXiv:2104.01725v2](https://arxiv.org/html/2104.01725v2)
