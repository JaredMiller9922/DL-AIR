# Professor Vs Repo Waveform Demo

This note isolates the waveform and noise assumptions that make the professor's demo easier to separate than the repo's default notebook experiments.

## 1. Carrier Separation

### Professor single-channel demo uses different carriers

File: `RF_SourseSeparate_finalV2.m`

```text
40: [x1,I1,Q1] = RF_qpsk_mod(s1, fs, fc, T);
41: [x2,I2,Q2] = RF_qpsk_mod(s2, fs, 2*fc, T);
52: s1h = qpsk_demod(x1_hat,fs,fc,T);
53: s2h = qpsk_demod(x2_hat,fs,2*fc,T);
```

What this means:

- Source 1 is transmitted at `fc`.
- Source 2 is transmitted at `2*fc`.
- Recovery also demodulates them with different carriers.
- This makes the one-channel professor setup much easier because the sources are already frequency-distinct.

### Professor four-antenna code uses the same carrier

File: `RF_gen_4ant_signal.m`

```text
24:     x1 = qpsk_mod(s1, fs, fc, T);
25:     x2 = qpsk_mod(s2, fs, fc, T);
```

File: `train_qpsk_mimo_separator.m`

```text
26:     x1 = qpsk_mod(s1, fs, fc, T);
27:     x2 = qpsk_mod(s2, fs, fc, T);
```

So the professor does use same-carrier sources in the four-antenna case.

### Repo default notebook experiments do not give the two sources different carriers

File: `utils/data_utils/generator.py`

```text
194:         s_soi, s_soi_symbols, soi_meta = self.generate_source(source_a_cfg, message=soi_message)
195:         s_int, s_int_symbols, int_meta = self.generate_source(source_b_cfg)
197:         s_int = self._apply_timing_offset(s_int, mix_cfg.timing_offset)
198:         s_int = self._apply_carrier_phase_mismatch(
199:             s_int,
200:             carrier_offset=mix_cfg.carrier_offset,
201:             phase_mismatch_deg=mix_cfg.phase_mismatch_deg,
202:         )
```

File: `config.py`

```text
72:     carrier_offset: float = 0.0
74:     phase_mismatch_deg: float = 0.0
```

What this means:

- By default, the repo generates both sources in the same baseband framework.
- Source B can be shifted only if `carrier_offset` or `phase_mismatch_deg` are explicitly changed.
- In the default notebook experiments, they are both zero, so there is no professor-style carrier split.

## 2. Samples Per Symbol

### Professor demodulates with 100 samples per symbol

File: `qpsk_demod.m`

```text
20: %% Step 2: Low-pass filtering (simple integration per symbol)
21: samples_per_symbol = fs*T;
22: num_symbols = floor(length(x_t)/samples_per_symbol);
27: for k = 1:num_symbols
28:     idx = (k-1)*samples_per_symbol + (1:samples_per_symbol);
31:     I_symbols(k) = sum(I_t(idx));
32:     Q_symbols(k) = sum(Q_t(idx));
```

File: `RF_SourseSeparate_finalV2.m`

```text
27: %     report = RF_SourseSeparate_finalV2('net',net1,100,100,2,1,1,1);
```

Since the example call uses `fs=100` and `T=1`, the professor integrates over `100` samples per symbol.

### Repo notebook tests use 2 samples per symbol

File: `config.py`

```text
39:     samples_per_symbol: int = 2
```

File: `utils/model_utils/symbol_utils.py`

```text
28:     # simplest sampling choice: every sps samples
30:     sym_samples = mf[::sps][:n_symbols]
```

What this means:

- The professor gets a large integration/processing gain before symbol decisions.
- The repo notebook experiments recover symbols after only `2` samples per symbol.
- This makes the repo much more sensitive to additive noise.

## 3. Noise Model And Normalization

### Repo normalizes the waveform power before mixing

File: `utils/data_utils/generator.py`

```text
360:         if cfg.normalize_power:
361:             shaped = self._normalize_complex_power(shaped)

510:     def _normalize_complex_power(self, x: np.ndarray) -> np.ndarray:
511:         p = np.mean(np.abs(x) ** 2) + 1e-12
512:         return x / np.sqrt(p)
```

This means each source waveform is scaled to approximately unit complex-sample power.

### Repo adds complex noise to the baseband mixture

File: `utils/data_utils/generator.py`

```text
377:         noise = (
378:             self.rng.standard_normal(signal.shape)
379:             + 1j * self.rng.standard_normal(signal.shape)
380:         ) / np.sqrt(2.0)
382:         if noise_cfg.sigma2 is not None:
383:             noise *= np.sqrt(noise_cfg.sigma2)
```

And that noise is added directly to the complex baseband signal:

```text
221:         if noise_cfg.enabled:
222:             noise = self.generate_noise(signal, noise_cfg)
223:             mixture = signal + noise
```

What this means:

- Repo noise hits both the real and imaginary parts of each IQ sample.
- `sigma2` is the variance of the complex noise sample, so each of `I` and `Q` gets `sigma2/2` variance.
- Because the sources are normalized near unit power, `sigma2=1` is already a strong noise level.

### Professor adds real-valued noise to the incoming bandpass wave

File: `RF_SourseSeparate_finalV2.m`

```text
43: noise = sqrt(sigma2)*randn(1,len_x1);
44: mix = x1 + alpha*x2 + noise;
```

What this means:

- The professor adds noise only to the real-valued incoming bandpass waveform.
- The bandpass waveform is then demodulated and integrated over `100` samples per symbol.
- So the same numeric `sigma2` does not mean the same effective difficulty as it does in the repo.

## 4. Professor Four-Antenna Random Delay Decorrelation

### Random delays are generated separately for each source and each antenna

File: `RF_gen_4ant_signal.m`

```text
32:     % Generate random delays for each antenna (in samples)
33:     maxDelay = round(0.1 * fs); % small delays
34:     delays1 = randi([0 maxDelay], 1, numAntennas);
35:     delays2 = randi([0 maxDelay], 1, numAntennas);
```

### Those delays are applied per antenna

File: `RF_gen_4ant_signal.m`

```text
40:     for m = 1:numAntennas
42:         d1 = delays1(m);
43:         d2 = delays2(m);
45:         x1_shift = [zeros(1,d1), x1(1:end-d1)];
46:         x2_shift = [zeros(1,d2), x2(1:end-d2)];
48:         X_multi(m,:) = x1_shift + x2_shift;
```

Why this matters:

- Each antenna sees a differently delayed version of each source.
- That decorrelates the antenna observations and makes the sources easier to separate.
- This is much more diverse than the repo's small deterministic phase-ramp array model.

## 5. Why The Professor's Alpha And Sigma2 Behave Differently

### Why alpha matters more in the professor's single-channel results

- In the professor's one-channel code, the sources live on different carriers (`fc` and `2*fc`).
- Changing `alpha` directly changes how strong the second carrier appears in the mixture.
- Because the demodulator already knows which carrier to look for, alpha has a strong visible effect.

### Why sigma2 is more abrupt in the repo notebook experiments

- The repo normalizes waveforms to unit power.
- The notebook experiments commonly use only `2` samples per symbol.
- Complex noise is added directly to the baseband IQ samples.
- So `sigma2=1` is already a low-SNR regime in the repo, while it is much less destructive in the professor's `100`-sample integrated waveform setup.

## 6. Separate Notebook For Normalized-Noise Analysis

See:

- `notebooks/analysis/normalized_noise_waveform_analysis.ipynb`

That notebook uses the repo generator without changing core behavior and shows:

- how to define noise power as a normalized fraction of waveform power,
- a surf plot over `alpha` and normalized `sigma2`, and
- 2D alpha sweeps at several normalized noise levels.
