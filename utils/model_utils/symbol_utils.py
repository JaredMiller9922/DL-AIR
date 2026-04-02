import numpy as np

QPSK_POINTS = np.array([
    (1 + 1j) / np.sqrt(2),
    (-1 + 1j) / np.sqrt(2),
    (-1 - 1j) / np.sqrt(2),
    (1 - 1j) / np.sqrt(2),
], dtype=np.complex64)

def iq_channels_to_complex(iq_2ch: np.ndarray) -> np.ndarray:
    return iq_2ch[0] + 1j * iq_2ch[1]

def nearest_qpsk_symbols(samples: np.ndarray) -> np.ndarray:
    # samples: (N,) complex
    dists = np.abs(samples[:, None] - QPSK_POINTS[None, :]) ** 2
    idx = np.argmin(dists, axis=1)
    return QPSK_POINTS[idx]

def recover_symbols_from_waveform(
    wave: np.ndarray,
    rrc_taps: np.ndarray,
    sps: int,
    n_symbols: int,
) -> np.ndarray:
    # matched filter
    mf = np.convolve(wave, rrc_taps, mode="same")

    # simplest sampling choice: every sps samples
    # for your synthetic data this is often good enough to start
    sym_samples = mf[::sps][:n_symbols]

    # hard decisions to nearest QPSK constellation point
    decided = nearest_qpsk_symbols(sym_samples)
    return decided

def symbol_accuracy(pred_syms: np.ndarray, true_syms: np.ndarray) -> float:
    return np.mean(pred_syms == true_syms)

# Define the pulse shape
def rrc_taps(sps: int, beta: float, span_symbols: int) -> np.ndarray:
    N = span_symbols * sps
    t = np.arange(-N, N + 1, dtype=np.float64) / sps

    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + (4 * beta / np.pi)
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            taps[i] = (
                beta
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / (den + 1e-12)

    taps /= np.sqrt(np.sum(taps ** 2) + 1e-12)
    return taps