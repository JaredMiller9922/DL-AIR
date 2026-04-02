import numpy as np
from sklearn.decomposition import FastICA

def fastica_two_signals(mixture_iq: np.ndarray):
    """
    mixture_iq: shape (8, T), real I/Q channels for one frame
    returns: s1, s2, ica
    """
    X = mixture_iq.T  # sklearn wants (n_samples, n_features) => (T, 8)

    ica = FastICA(
        n_components=2,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        random_state=0,
    )

    S = ica.fit_transform(X)  # shape (T, 2)

    s1 = S[:, 0]
    s2 = S[:, 1]
    return s1, s2, ica