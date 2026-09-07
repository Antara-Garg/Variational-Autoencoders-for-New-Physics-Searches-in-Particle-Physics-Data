import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from datahandlingnew import train_df, val_df, test_df, label_col

# ---------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------
LATENT_DIM = 10
BETA = 10
N_EPOCHS = 70
PATIENCE = 7
BATCH_SIZE = 4096

# ---------------------------------------------------------------
# BASELINE (known-clean) FEATURES -- defined here, not imported
# ---------------------------------------------------------------
BASE_FEATURES = [
    "mj1", "mj2",
    "tau21j1", "tau21j2",
    "tau32j1", "tau32j2",
    "tau1j1", "tau2j1", "tau3j1",
    "tau1j2", "tau2j2", "tau3j2",
    "ptj1", "ptj2",
]

# candidate features to test one at a time, added on top of BASE_FEATURES
CANDIDATE_FEATURES = [
    "delta_eta", "delta_phi", "delta_R", "cos_theta_star",
    "mass_asym", "pt_asym", "system_pt", "HT",
    "tau31j1", "tau31j2",
]

MJ1_INDEX_IN_BASE = 0  # "mj1" is the first entry of BASE_FEATURES


# ---------------------------------------------------------------
# MODEL DEFINITION (same VAE structure used throughout)
# ---------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


# ---------------------------------------------------------------
# ONE FULL TRAIN + EVAL RUN, GIVEN A FEATURE LIST
# ---------------------------------------------------------------
def run_experiment(feature_cols):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[label_col].values

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()

    train_loader = DataLoader(TensorDataset(X_train_t, X_train_t),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, X_val_t),
                             batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch_x)
            loss = vae_loss(x_hat, batch_x, mu, logvar, BETA)
            loss.backward()
            optimizer.step()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                x_hat, mu, logvar = model(batch_x)
                loss = vae_loss(x_hat, batch_x, mu, logvar, BETA)
                val_running_loss += loss.item()
        val_loss = val_running_loss / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        x_hat_test, _, _ = model(X_test_t)
        anomaly_score = torch.mean((x_hat_test - X_test_t) ** 2, dim=1).numpy()

    auc = roc_auc_score(y_test, anomaly_score)

    bkg_mask = (y_test == 0)
    bkg_mass = X_test[bkg_mask, MJ1_INDEX_IN_BASE]
    bkg_scores = anomaly_score[bkg_mask]
    corr, p_value = pearsonr(bkg_mass, bkg_scores)

    return auc, corr, p_value


# ---------------------------------------------------------------
# SWEEP: baseline, then baseline + each candidate feature, one at a time
# ---------------------------------------------------------------
print(f"{'Feature set':40s} {'AUC':>8s} {'corr(mj1)':>12s} {'p-value':>12s}")
print("-" * 76)

auc, corr, p = run_experiment(BASE_FEATURES)
print(f"{'BASELINE (14 features)':40s} {auc:8.4f} {corr:12.4f} {p:12.2e}")

for feat in CANDIDATE_FEATURES:
    feature_cols = BASE_FEATURES + [feat]
    auc, corr, p = run_experiment(feature_cols)
    print(f"{'baseline + ' + feat:40s} {auc:8.4f} {corr:12.4f} {p:12.2e}")

print("\nDone. Features with |corr| well below ~0.2 and AUC >= baseline are good candidates to keep.")