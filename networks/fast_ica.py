import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import FastICA as SkFastICA

class FastICABaseline(nn.Module):
    """
    ICA baseline that matches NN output shape.

    Input:  (B, C, T)
    Output: (B, 4, T)
    """

    def __init__(self, random_state=0, max_iter=20000):
        super().__init__()
        self.random_state = random_state
        self.max_iter = max_iter

    def forward(self, x):
        B, C, T = x.shape

        # Must detach so pytorch doesn't attempt to do gradients
        x_np = x.detach().cpu().numpy()
        outs = []

        # Our DataLoader will use batching so we need to loop through every item in the batch
        for b in range(B):
            sample = x_np[b]      # (C, T)

            # Standardize each sample
            sample = sample - sample.mean(axis=1, keepdims=True)
            sample = sample / (sample.std(axis=1, keepdims=True) + 1e-8)

            X = sample.T          # (T, C)

            ica = SkFastICA(
                n_components=4,
                algorithm="parallel",
                whiten="unit-variance",
                fun="logcosh",
                random_state=self.random_state,
                max_iter=self.max_iter,
            )

            # Reshape data to be in desired output format
            S = ica.fit_transform(X)   # (T, 4)
            print(f"FastICA iterations (sample {b}): {ica.n_iter_}")
            outs.append(S.T.astype(np.float32))  # (4, T)

        y = np.stack(outs, axis=0)  # (B, 4, T)
        return torch.from_numpy(y).to(x.device)