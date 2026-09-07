import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from datahandlingnew import X_train, X_val, X_test, Y_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

MJ1_INDEX = 0

  
# 1. ENCODER
  
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


  
# 2. DECODER
  
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.net(z)


  
# 3. FULL VAE
  
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
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


  
# 4. DISTANCE CORRELATION (DisCo)
  
def distance_corr(x, y, eps=1e-9):
    x = x.view(-1, 1)
    y = y.view(-1, 1)

    a = torch.cdist(x, x, p=2)
    b = torch.cdist(y, y, p=2)

    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

    dCov2 = (A * B).mean()
    dVarX2 = (A * A).mean()
    dVarY2 = (B * B).mean()

    dCorr = dCov2 / torch.sqrt(dVarX2 * dVarY2 + eps)
    return dCorr


  
# 5. LOSS FUNCTION: reconstruction + KL divergence + DisCo penalty
  
def vae_loss(x_hat, x, mu, logvar, mj1_batch, beta1, beta2, lambda_disco, disco_sample_size=512):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # per-event reconstruction error = the anomaly score, computed
    # inside the loss so it can be penalized for correlating with mj1
    per_event_score = torch.mean((x_hat - x) ** 2, dim=1)

    # subsample for the DisCo computation specifically pairwise distance
    n = per_event_score.size(0)
    if n > disco_sample_size:
        idx = torch.randperm(n, device=per_event_score.device)[:disco_sample_size]
        disco_penalty = distance_corr(per_event_score[idx], mj1_batch[idx])
    else:
        disco_penalty = distance_corr(per_event_score, mj1_batch)

    loss = beta1 * recon_loss + beta2 * kl_div + lambda_disco * disco_penalty
    return loss, recon_loss, kl_div, disco_penalty


  
# 6. DATA LOADERS
  
X_train_t = torch.from_numpy(X_train).float().to(device)
X_val_t = torch.from_numpy(X_val).float().to(device)
X_test_t = torch.from_numpy(X_test).float().to(device)

train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=4096, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, X_val_t), batch_size=4096, shuffle=False)

  
# 7. TRAINING SETUP
  
latent_dim = 10

input_dim = X_train.shape[1]
model = VAE(input_dim=input_dim, latent_dim= latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 70
patience = 12
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

BETA1 = 1
BETA2 = 10
LAMBDA_DISCO = 100000 

train_losses, val_losses = [], []

print("\nDisCo: ", LAMBDA_DISCO)
print("Latend Dim: ", latent_dim)

  
# 8. TRAINING LOOP
  
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        x_hat, mu, logvar = model(batch_x)
        mj1_batch = batch_x[:, MJ1_INDEX]
        loss, recon_loss, kl_div, disco = vae_loss(
            x_hat, batch_x, mu, logvar, mj1_batch, BETA1, BETA2, LAMBDA_DISCO
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch_x, _ in val_loader:
            x_hat, mu, logvar = model(batch_x)
            mj1_batch = batch_x[:, MJ1_INDEX]
            loss, _, _, _ = vae_loss(
                x_hat, batch_x, mu, logvar, mj1_batch, BETA1, BETA2, LAMBDA_DISCO
            )
            val_running_loss += loss.item()
    val_loss = val_running_loss / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f} - disco: {disco.item():.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "vae_weights.pt")

  
# 9. LOSS CURVE
  
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("VAE loss (recon + KL + DisCo)")
plt.legend()
plt.title("VAE training curve (with DisCo)")
plt.savefig("vae_loss_curve.png")
plt.close()

  
# 10. ANOMALY SCORE ON TEST SET
  
model.eval()
with torch.no_grad():
    x_hat_test, mu_test, logvar_test = model(X_test_t)
    anomaly_score = torch.mean((x_hat_test - X_test_t) ** 2, dim=1).cpu().numpy()

auc = roc_auc_score(Y_test, anomaly_score)
print(f"\nTest AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(Y_test, anomaly_score)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("Background efficiency (FPR)")
plt.ylabel("Signal efficiency (TPR)")
plt.legend()
plt.title("VAE ROC curve (with DisCo)")
plt.savefig("vae_roc_curve.png")
plt.close()

  
# 11. ANOMALY SCORE DISTRIBUTION
  
bkg_scores = anomaly_score[Y_test == 0]
sig_scores = anomaly_score[Y_test == 1]

plt.figure()
plt.hist(bkg_scores, bins=50, alpha=0.5, density=True, label="background (true)")
plt.hist(sig_scores, bins=50, alpha=0.5, density=True, label="signal (true)")
plt.xlabel("Anomaly score (reconstruction MSE)")
plt.ylabel("Normalized events")
plt.legend()
plt.title("VAE anomaly score by true label (with DisCo)")
plt.savefig("vae_score_distribution.png")
plt.close()


  
# 12. MASS-SCULPTING CHECK (background only)
  
from scipy.stats import pearsonr

bkg_mask = (Y_test == 0)
bkg_scores = anomaly_score[bkg_mask]
bkg_mass = X_test[bkg_mask, MJ1_INDEX]

corr, p_value = pearsonr(bkg_mass, bkg_scores)
print(f"\nPearson corr: {corr:.4f}")
print(f"p-value: {p_value:.4e}")

plt.figure()
plt.scatter(bkg_mass, bkg_scores, s=2, alpha=0.3)
plt.xlabel("mj1 (standardized)")
plt.ylabel("Anomaly score (reconstruction MSE)")
plt.title(f"Mass-sculpting check (background only), corr = {corr:.3f}")
plt.savefig("vae_mass_sculpting_check.png")
plt.close()