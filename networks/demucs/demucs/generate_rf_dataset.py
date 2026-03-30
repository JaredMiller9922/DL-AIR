import numpy as np
from pathlib import Path
import argparse


#########################################
# RF SIGNAL GENERATORS
#########################################

def generate_bpsk(n, fs, fc):

    bits = np.random.randint(0, 2, n)
    symbols = 2*bits - 1

    t = np.arange(n) / fs

    carrier = np.exp(1j * 2 * np.pi * fc * t)

    return symbols * carrier


def generate_qpsk(n, fs, fc):

    bits = np.random.randint(0, 4, n)

    mapping = {
        0: 1+1j,
        1: -1+1j,
        2: -1-1j,
        3: 1-1j
    }

    symbols = np.array([mapping[b] for b in bits]) / np.sqrt(2)

    t = np.arange(n) / fs

    carrier = np.exp(1j * 2 * np.pi * fc * t)

    return symbols * carrier


def generate_fm(n, fs, fc):

    t = np.arange(n) / fs

    mod = np.sin(2*np.pi*1000*t)

    phase = 2*np.pi*fc*t + 5*mod

    return np.exp(1j * phase)


def generate_am(n, fs, fc):

    t = np.arange(n) / fs

    mod = 0.5*np.sin(2*np.pi*1000*t)

    carrier = np.exp(1j * 2*np.pi*fc*t)

    return (1 + mod) * carrier


#########################################
# RANDOM SIGNAL FACTORY
#########################################

# def generate_signal(n, fs):

#     fc = np.random.uniform(-fs/4, fs/4)

#     mod_type = np.random.choice(["bpsk", "qpsk", "fm", "am"])

#     if mod_type == "bpsk":
#         sig = generate_bpsk(n, fs, fc)

#     elif mod_type == "qpsk":
#         sig = generate_qpsk(n, fs, fc)

#     elif mod_type == "fm":
#         sig = generate_fm(n, fs, fc)

#     elif mod_type == "am":
#         sig = generate_am(n, fs, fc)

#     amp = np.random.uniform(0.3, 1.0)

#     phase = np.exp(1j * np.random.uniform(0, 2*np.pi))

#     return amp * phase * sig


def generate_signal(n, fs):

    fc = np.random.uniform(-fs/4, fs/4)

    mod_type = np.random.choice(["bpsk","qpsk","fm","am"])

    if mod_type == "bpsk":
        sig = generate_bpsk(n, fs, fc)

    elif mod_type == "qpsk":
        sig = generate_qpsk(n, fs, fc)

    elif mod_type == "fm":
        sig = generate_fm(n, fs, fc)

    elif mod_type == "am":
        sig = generate_am(n, fs, fc)

    sig = burstify(sig, n)

    sig = apply_fading(sig)

    sig = apply_doppler(sig, fs)

    amp = np.random.uniform(0.2, 1.0)

    phase = np.exp(1j*np.random.uniform(0,2*np.pi))

    return amp * phase * sig


#########################################
# MIXTURE CREATION
#########################################

def generate_sample(n, fs, num_sources):

    sources = []

    for _ in range(num_sources):
        sources.append(generate_signal(n, fs))

    sources = np.stack(sources)

    mixture = sources.sum(axis=0)

    noise_power = np.random.uniform(0.001, 0.01)

    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n) + 1j*np.random.randn(n)
    )

    mixture += noise

    return mixture, sources


def apply_fading(sig):

    fading = np.cumsum(np.random.randn(len(sig)) * 0.0001)

    fading = np.exp(1j * fading)

    return sig * fading


def apply_doppler(sig, fs):

    drift = np.random.uniform(-50, 50)

    t = np.arange(len(sig)) / fs

    return sig * np.exp(1j * 2*np.pi*drift*t)


def burstify(sig, n):

    start = np.random.randint(0, n//2)

    length = np.random.randint(n//4, n)

    out = np.zeros(n, dtype=np.complex64)

    end = min(start + length, n)

    out[start:end] = sig[:end-start]

    return out




#########################################
# DATASET WRITER
#########################################

def write_dataset(root, samples, length, fs, sources):

    root = Path(root)

    root.mkdir(parents=True, exist_ok=True)

    for i in range(samples):

        sample_dir = root / f"sample{i:05d}"
        sample_dir.mkdir(exist_ok=True)

        mixture, srcs = generate_sample(length, fs, sources)

        # mixture.astype(np.complex64).tofile(sample_dir / "mixture.iq")

        # for j in range(sources):

        #     srcs[j].astype(np.complex64).tofile(
        #         sample_dir / f"source{j+1}.iq"
        #     )

        # Change .iq to .wav
        mixture.astype(np.complex64).tofile(sample_dir / "mixture.wav")
        
        for j in range(sources):
            srcs[j].astype(np.complex64).tofile(sample_dir / f"source{j+1}.wav")

        if i % 100 == 0:
            print("generated", i)


#########################################
# CLI
#########################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--fs", type=float, default=1e6)

    args = parser.parse_args()

    write_dataset(
        args.out,
        args.samples,
        args.length,
        args.fs,
        args.sources
    )


if __name__ == "__main__":
    main()