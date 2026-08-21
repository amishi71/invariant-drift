"""AXOL1TL-proxy variational autoencoder.

CMS's real AXOL1TL trigger runs on L1 trigger primitives (calorimeter
regions, L1 muon/jet objects) inside the Level-1 hardware -- those raw
primitives are not part of public CMS/LHC Open Data. As agreed with the
advisor (see docs/ADVISOR_NOTES.md), this project instead trains a VAE on
*offline-reconstructed* object- and event-level features that are the
closest public analog: jet kinematics, MET, HT, and object multiplicities
from NanoAOD-like events. The VAE's reconstruction error stands in for
AXOL1TL's anomaly score. This is a proxy, not a reproduction of AXOL1TL
itself -- the calibration-drift-detection methodology (everything in
src/detectors, src/residual.py, src/conformal) is what's actually being
evaluated, and it is agnostic to which upstream anomaly-score model
produces the scalar it watches.

Trained on background-only ("Zero-Bias control") events only, same
convention as CICADA/AXOL1TL themselves and as the calibration burn-in
used everywhere else in this project: the model should reconstruct nominal
events well and nominal-but-different-conditions events less well, and the
reconstruction error is the anomaly score fed into residual.py.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


DEFAULT_FEATURE_NAMES = [
    "jet1_pt", "jet1_eta", "jet1_phi",
    "jet2_pt", "jet2_eta", "jet2_phi",
    "n_jet", "met_pt", "met_phi", "ht",
    "n_muon", "n_electron",
]


class ProxyVAE(nn.Module):
    """Small fully-connected VAE. Object-level features here are a flat
    vector (jet kinematics, MET, HT, multiplicities), not a calorimeter
    image, so a plain MLP encoder/decoder is the right architecture --
    CICADA's convolutional design (built for a regular calo-tower grid)
    doesn't apply to this feature representation, which is exactly why
    CICADA itself was dropped from scope in favor of the AXOL1TL-style
    object-based proxy (see docs/ADVISOR_NOTES.md).
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (32, 16), latent_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_layers = []
        d = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

        dec_layers = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, kl_weight: float = 1.0):
    """Standard VAE ELBO: reconstruction MSE + KL(q(z|x) || N(0,I))."""
    recon_loss = torch.mean(torch.sum((recon - x) ** 2, dim=1))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + kl_weight * kl, recon_loss, kl


@dataclass
class FeatureScaler:
    """Frozen z-score scaler fit on burn-in background features (same
    frozen-reference discipline as everything else in this project)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureScaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)  # guard against constant columns
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def train_proxy_vae(
    background_features: np.ndarray,
    hidden_dims: Sequence[int] = (32, 16),
    latent_dim: int = 6,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    kl_weight: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
):
    """Fits the scaler and trains the VAE on burn-in background-only
    (Zero-Bias-like) object features. Returns (model, scaler, history).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    X = np.asarray(background_features, dtype=np.float64)
    scaler = FeatureScaler.fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    model = ProxyVAE(input_dim=Xs.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xs.shape[0]
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(Xs[idx])
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss, recon_loss, kl = vae_loss(recon, xb, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        epoch_loss /= n
        history.append(epoch_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [proxy_vae] epoch {epoch:3d}  loss={epoch_loss:.4f}")

    model.eval()
    return model, scaler, history


@torch.no_grad()
def anomaly_score(model: ProxyVAE, scaler: FeatureScaler, features: np.ndarray) -> np.ndarray:
    """Per-event reconstruction error (mean squared error in scaled feature
    space) -- the anomaly score fed into residual.py as `score`.

    Uses the VAE's mean reconstruction (decode(mu), no sampling noise) so
    the score is deterministic given the frozen model -- important for a
    monitoring signal, where you don't want detector noise conflated with
    stochastic-sampling noise from the scorer itself.
    """
    X = np.asarray(features, dtype=np.float64)
    Xs = torch.from_numpy(scaler.transform(X).astype(np.float32))
    mu, logvar = model.encode(Xs)
    recon = model.decode(mu)
    err = torch.mean((recon - Xs) ** 2, dim=1)
    return err.numpy().astype(np.float64)


@torch.no_grad()
def anomaly_score_one(model: ProxyVAE, scaler: FeatureScaler, feature_vector: np.ndarray) -> float:
    """Convenience scalar version for event-by-event streaming use."""
    return float(anomaly_score(model, scaler, np.asarray(feature_vector).reshape(1, -1))[0])
