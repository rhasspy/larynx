import numpy as np


def mel_basis(sr, n_fft, n_mels=80, fmin=0.0, fmax=None, dtype=np.float32):

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)  # pylint: disable=no-member

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels)


def fft_frequencies(sr=22050, n_fft=2048):
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels):
    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def stft(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array(
        [
            np.fft.rfft(window * x[i : i + fft_size])
            for i in range(0, len(x) - fft_size, hopsamp)
        ]
    )


def istft(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.
    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.
    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices * hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n, i in enumerate(range(0, len(x) - fft_size, hopsamp)):
        x[i : i + fft_size] += window * np.real(np.fft.irfft(X[n]))
    return x


def inverse(magnitude, phase):
    recombine_magnitude_phase = np.concatenate(
        [magnitude * np.cos(phase), magnitude * np.sin(phase)], axis=1
    )

    x_org = recombine_magnitude_phase
    n_b, n_f, n_t = x_org.shape  # pylint: disable=unpacking-non-sequence
    x = np.empty([n_b, n_f // 2, n_t], dtype=np.complex64)
    x.real = x_org[:, : n_f // 2]
    x.imag = x_org[:, n_f // 2 :]
    inverse_transform = []
    for y in x:
        y_ = istft(y.T, fft_size=1024, hopsamp=256)
        inverse_transform.append(y_[None, :])

    inverse_transform = np.concatenate(inverse_transform, 0)

    return inverse_transform


def transform(input_data):
    x = input_data
    real_part = []
    imag_part = []
    for y in x:
        y_ = stft(y, fft_size=1024, hopsamp=256).T
        real_part.append(y_.real[None, :, :])  # pylint: disable=unsubscriptable-object
        imag_part.append(y_.imag[None, :, :])  # pylint: disable=unsubscriptable-object
    real_part = np.concatenate(real_part, 0)
    imag_part = np.concatenate(imag_part, 0)

    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
    phase = np.arctan2(imag_part.data, real_part.data)

    return magnitude, phase
