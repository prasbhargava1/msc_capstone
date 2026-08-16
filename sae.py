"""
Sparse autoencoder: encoder -> ReLU -> sparse features -> decoder ->
reconstruction. See Methodology / Sparse Autoencoder Training for the
full explanation of the shared bias and the loss function.
"""

import os

import numpy as np
import torch
import torch.nn as nn

import config


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor=None):
        super().__init__()
        expansion_factor = expansion_factor or config.SAE_EXPANSION_FACTOR
        dict_size = input_dim * expansion_factor

        self.W_enc = nn.Parameter(torch.randn(input_dim, dict_size) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.randn(dict_size, input_dim) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))  # shared: used in both encode and decode

        self.normalize_decoder()

    def encode(self, x):
        # b_dec is subtracted here (not just added at the end) so the
        # encoder only has to represent deviation from the "typical"
        # activation, not the typical activation itself
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, features):
        return features @ self.W_dec + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        x_hat = self.decode(features)
        return features, x_hat

    @torch.no_grad()
    def normalize_decoder(self):
        # keep each feature's output direction at unit norm, so a
        # feature's activation strength alone controls how much it
        # contributes to the reconstruction
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def sae_loss(x, x_hat, features, l1_coeff=None):
    l1_coeff = l1_coeff if l1_coeff is not None else config.SAE_L1_COEFF
    recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
    sparsity_loss = features.abs().sum(dim=-1).mean()
    return recon_loss + l1_coeff * sparsity_loss, recon_loss.item(), sparsity_loss.item()


def train_sae(activations, save_path, device=None, verbose=True):
    """
    Train one SAE on a [n_examples, hidden_dim] activation matrix.
    If save_path already has a checkpoint, load and return that instead
    of retraining - this is what makes re-running the pipeline cheap.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(input_dim).to(device)

    if os.path.exists(save_path):
        print(f"  [cache hit] loading SAE from {save_path}")
        sae.load_state_dict(torch.load(save_path, map_location=device))
        return sae

    print(f"  [training] SAE -> {save_path}")

    x = torch.tensor(activations, dtype=torch.float32, device=device)
    # initialise b_dec at the data mean - a much better starting point
    # than zero, since that is roughly what it should converge to anyway
    sae.b_dec.data = x.mean(dim=0)

    n = x.shape[0]
    n_val = max(1, int(0.1 * n))
    perm = torch.randperm(n)
    val_x, train_x = x[perm[:n_val]], x[perm[n_val:]]

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.SAE_LR)

    best_val_loss = float("inf")
    patience_left = config.SAE_EARLY_STOP_PATIENCE
    best_state = None

    for epoch in range(config.SAE_EPOCHS):
        sae.train()
        perm = torch.randperm(train_x.shape[0])
        for i in range(0, train_x.shape[0], config.SAE_BATCH_SIZE):
            batch = train_x[perm[i : i + config.SAE_BATCH_SIZE]]
            features, x_hat = sae(batch)
            loss, recon, sparsity = sae_loss(batch, x_hat, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        sae.eval()
        with torch.no_grad():
            _, val_x_hat = sae(val_x)
            val_loss = ((val_x - val_x_hat) ** 2).sum(dim=-1).mean().item()

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch}: val recon loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = config.SAE_EARLY_STOP_PATIENCE
            best_state = {k: v.clone() for k, v in sae.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"    early stopping at epoch {epoch}")
                break

    if best_state is not None:
        sae.load_state_dict(best_state)

    torch.save(sae.state_dict(), save_path)
    return sae
