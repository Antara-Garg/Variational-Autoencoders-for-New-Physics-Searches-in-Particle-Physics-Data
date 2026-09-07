# Variational-Autoencoders-for-New-Physics-Searches-in-Particle-Physics-Data
Unsupervised anomaly detection for model-independent new physics searches, using the LHC Olympics 2020 R&D dataset.

## Data & Features

- **Dataset**: LHC Olympics 2020 R&D dataset (high-level jet features), a public benchmark for anomaly detection at the LHC.
- **Features**: per-jet mass, n-subjettiness ratios (τ21, τ32), transverse momentum, and other substructure observables; deliberately excluded dijet invariant mass (mjj) as a training input, since that's the variable used to define the signal region.
- **Setup**: model trained only on background-like events from a sideband region in mjj; evaluated on the signal region, where true labels are used only for evaluation.

## Methodology

1. **Baseline models**: a plain autoencoder and a variational autoencoder (VAE), both trained to reconstruct background events, using reconstruction error as the anomaly score.
2. **Mass-sculpting check**: correlating the anomaly score with jet mass, on background-only events, to test whether the model was learning genuine substructure or secretly exploiting mass as a shortcut.
3. **Finding**: certain engineered features and an oversized latent dimension caused the anomaly score to correlate strongly with jet mass (Pearson r up to ~0.48)
4. **Fix**: added a **distance correlation (DisCo) regularization** term to the VAE's loss function, directly penalizing correlation between the anomaly score and jet mass during training (Kasieczka & Shih, [arXiv:2001.05310](https://arxiv.org/abs/2001.05310)).

## Results

| Model | AUC | Correlation with mj1 (background) |
|---|---|---|
| Autoencoder (baseline) | 0.76 | — |
| VAE (uncorrected) | up to 0.80 | up to 0.48 (leaking) |
| **VAE + DisCo (corrected)** | **0.73–0.75** | **~0.10–0.14** |

## Next Steps

- Derive richer substructure features (energy correlation functions, jet width) from raw particle-level data rather than high-level approximations.
- Explore hybrid approaches (VAE encoder + classical detector, e.g. Isolation Forest) and one-class neural networks (OCNN).
- Extend to graph-based models operating on jet constituents directly.
