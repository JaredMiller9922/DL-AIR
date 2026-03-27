import numpy as np

def complex_to_2ch(x: np.ndarray) -> np.ndarray:
    """
    Convert complex vector of shape (T,) to real array of shape (2, T):
      [I;
       Q]
    """
    if x.ndim != 1:
        raise ValueError(f"Expected shape (T,), got {x.shape}")
    return np.stack([x.real, x.imag], axis=0).astype(np.float32)


def complex_matrix_to_iq_channels(X: np.ndarray) -> np.ndarray:
    """
    Convert complex matrix of shape (C, T) into real I/Q channels of shape (2*C, T).

    Example:
      (4, T) complex -> (8, T) real
      channel order: [ch0_I, ch0_Q, ch1_I, ch1_Q, ...]
    """
    if X.ndim != 2:
        raise ValueError(f"Expected shape (C, T), got {X.shape}")

    C, T = X.shape
    out = np.zeros((2 * C, T), dtype=np.float32)

    for c in range(C):
        out[2 * c] = X[c].real
        out[2 * c + 1] = X[c].imag

    return out


def iq_channels_to_complex_matrix(X_iq: np.ndarray) -> np.ndarray:
    """
    Convert real I/Q channels of shape (2*C, T) back to complex matrix of shape (C, T).
    """
    if X_iq.ndim != 2:
        raise ValueError(f"Expected shape (2*C, T), got {X_iq.shape}")
    if X_iq.shape[0] % 2 != 0:
        raise ValueError("First dimension must be even (I/Q pairs).")

    C2, T = X_iq.shape
    C = C2 // 2
    X = np.zeros((C, T), dtype=np.complex64)

    for c in range(C):
        X[c] = X_iq[2 * c] + 1j * X_iq[2 * c + 1]

    return X


def stacked_sources_to_iq(source_a: np.ndarray, source_b: np.ndarray) -> np.ndarray:
    """
    source_a, source_b: complex arrays of shape (T,)
    returns: real array of shape (4, T)
      [a_I, a_Q, b_I, b_Q]
    """
    a = complex_to_2ch(source_a)
    b = complex_to_2ch(source_b)
    return np.concatenate([a, b], axis=0).astype(np.float32)
