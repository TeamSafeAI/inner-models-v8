"""
encoder.py -- Convert real-world signals to neuron currents.

Agnostic to modality. Takes time-series data, maps channels to neurons,
produces external_I arrays for Brain.tick().

Encoding: rate coding. Higher signal value = more current = more firing.
The brain figures out what the patterns mean.

Usage:
    from engine.encoder import SignalEncoder

    # From audio file (requires scipy for wav reading)
    enc = SignalEncoder.from_audio('sound.wav', n_channels=32, tick_rate=1000)

    # From raw time-series (channels x samples)
    enc = SignalEncoder.from_array(data, tick_rate=1000)

    # Map channels to neuron indices
    enc.map_channels(neuron_indices)  # list of lists, one per channel

    # In tick loop:
    I_ext = enc.get_current(tick)  # returns array of length n_neurons
"""
import numpy as np


class SignalEncoder:
    """Converts multi-channel time-series to neuron currents per tick."""

    def __init__(self, channels, source_rate, tick_rate=1000):
        """
        Args:
            channels: ndarray (n_channels, n_samples) — signal per channel
            source_rate: sample rate of the source signal (Hz)
            tick_rate: brain ticks per second (default 1000 = 1ms per tick)
        """
        self.channels = np.asarray(channels, dtype=np.float64)
        self.n_channels = self.channels.shape[0]
        self.n_samples = self.channels.shape[1]
        self.source_rate = source_rate
        self.tick_rate = tick_rate
        self.channel_map = None  # set by map_channels()
        self.n_neurons = 0

        # Normalize GLOBALLY to [0, 1] — preserves relative energy between channels
        # (a loud frequency band should drive neurons harder than a quiet one)
        mn = self.channels.min()
        mx = self.channels.max()
        if mx - mn > 1e-10:
            self.channels = (self.channels - mn) / (mx - mn)
        else:
            self.channels = np.zeros_like(self.channels)

        # Duration in seconds
        self.duration = self.n_samples / self.source_rate
        self.total_ticks = int(self.duration * self.tick_rate)

    @classmethod
    def from_array(cls, data, source_rate, tick_rate=1000):
        """Create from raw array (n_channels, n_samples)."""
        return cls(data, source_rate, tick_rate)

    @classmethod
    def from_audio(cls, path, n_channels=32, tick_rate=1000):
        """Create from audio file using frequency decomposition.

        Splits audio into n_channels frequency bands (like the cochlea).
        Each band's energy envelope becomes one channel.

        Args:
            path: path to .wav file
            n_channels: number of frequency bands (default 32)
            tick_rate: brain ticks per second
        """
        from scipy.io import wavfile
        sample_rate, raw = wavfile.read(path)

        # Mono
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        raw = raw.astype(np.float64)

        # Normalize to [-1, 1]
        mx = np.abs(raw).max()
        if mx > 0:
            raw = raw / mx

        # STFT-based frequency decomposition
        # Window size: ~25ms (standard for audio), hop: ~10ms
        win_samples = int(0.025 * sample_rate)
        hop_samples = int(0.010 * sample_rate)
        if win_samples < 64:
            win_samples = 64
        if hop_samples < 1:
            hop_samples = 1

        # Compute spectrogram
        n_frames = max(1, (len(raw) - win_samples) // hop_samples + 1)
        channels = np.zeros((n_channels, n_frames))

        window = np.hanning(win_samples)
        n_fft = win_samples
        freqs_per_bin = sample_rate / n_fft

        # Logarithmic frequency bands (like cochlea — more resolution at low freq)
        min_freq = 80.0
        max_freq = min(sample_rate / 2.0, 16000.0)
        band_edges = np.logspace(
            np.log10(min_freq), np.log10(max_freq), n_channels + 1
        )

        for frame_idx in range(n_frames):
            start = frame_idx * hop_samples
            segment = raw[start:start + win_samples]
            if len(segment) < win_samples:
                segment = np.pad(segment, (0, win_samples - len(segment)))
            windowed = segment * window
            spectrum = np.abs(np.fft.rfft(windowed))

            for ch in range(n_channels):
                lo_bin = int(band_edges[ch] / freqs_per_bin)
                hi_bin = int(band_edges[ch + 1] / freqs_per_bin)
                lo_bin = max(0, min(lo_bin, len(spectrum) - 1))
                hi_bin = max(lo_bin + 1, min(hi_bin, len(spectrum)))
                channels[ch, frame_idx] = spectrum[lo_bin:hi_bin].mean()

        # The source_rate for the spectrogram frames
        frame_rate = sample_rate / hop_samples

        return cls(channels, frame_rate, tick_rate)

    @classmethod
    def from_tone(cls, freq=440.0, duration=1.0, n_channels=32, tick_rate=1000):
        """Create from a pure sine tone (for testing).

        Generates a synthetic audio signal and decomposes it.
        The channel corresponding to freq should light up.
        """
        sample_rate = 16000
        t = np.arange(int(sample_rate * duration)) / sample_rate
        raw = np.sin(2 * np.pi * freq * t)

        # Do the same decomposition as from_audio
        win_samples = int(0.025 * sample_rate)
        hop_samples = int(0.010 * sample_rate)
        n_frames = max(1, (len(raw) - win_samples) // hop_samples + 1)
        channels = np.zeros((n_channels, n_frames))

        window = np.hanning(win_samples)
        n_fft = win_samples
        freqs_per_bin = sample_rate / n_fft

        min_freq = 80.0
        max_freq = min(sample_rate / 2.0, 16000.0)
        band_edges = np.logspace(
            np.log10(min_freq), np.log10(max_freq), n_channels + 1
        )

        for frame_idx in range(n_frames):
            start = frame_idx * hop_samples
            segment = raw[start:start + win_samples]
            if len(segment) < win_samples:
                segment = np.pad(segment, (0, win_samples - len(segment)))
            windowed = segment * window
            spectrum = np.abs(np.fft.rfft(windowed))

            for ch in range(n_channels):
                lo_bin = int(band_edges[ch] / freqs_per_bin)
                hi_bin = int(band_edges[ch + 1] / freqs_per_bin)
                lo_bin = max(0, min(lo_bin, len(spectrum) - 1))
                hi_bin = max(lo_bin + 1, min(hi_bin, len(spectrum)))
                channels[ch, frame_idx] = spectrum[lo_bin:hi_bin].mean()

        frame_rate = sample_rate / hop_samples
        return cls(channels, frame_rate, tick_rate)

    def map_channels(self, neuron_indices, n_neurons=None):
        """Map encoder channels to brain neuron indices.

        Args:
            neuron_indices: list of lists. neuron_indices[ch] = [idx1, idx2, ...]
                           neurons to drive for channel ch.
            n_neurons: total neuron count in brain (for output array size).
                       If None, inferred from max index.
        """
        self.channel_map = neuron_indices
        if n_neurons is not None:
            self.n_neurons = n_neurons
        else:
            self.n_neurons = max(max(ch) for ch in neuron_indices if ch) + 1

    def get_current(self, tick, current_scale=10.0):
        """Get external current array for this brain tick.

        Args:
            tick: current brain tick number
            current_scale: max current to inject (default 10.0 mA).
                          Channel value 1.0 -> current_scale mA.

        Returns:
            numpy array of length n_neurons with current per neuron.
        """
        I = np.zeros(self.n_neurons, dtype=np.float64)

        if self.channel_map is None:
            return I

        # Map brain tick to source sample index
        source_time = tick / self.tick_rate  # seconds
        sample_idx = int(source_time * self.source_rate)

        if sample_idx >= self.n_samples:
            return I  # past end of signal

        for ch in range(self.n_channels):
            if ch >= len(self.channel_map):
                break
            val = self.channels[ch, sample_idx] * current_scale
            for nidx in self.channel_map[ch]:
                I[nidx] += val

        return I

    @property
    def finished(self):
        """True if we've run past the end of the signal."""
        return False  # caller decides when to stop

    def info(self):
        """Print encoder summary."""
        print(f"  Encoder: {self.n_channels} channels, {self.n_samples} frames")
        print(f"  Source rate: {self.source_rate:.0f} Hz, tick rate: {self.tick_rate} Hz")
        print(f"  Duration: {self.duration:.2f}s = {self.total_ticks} ticks")
        if self.channel_map:
            n_mapped = sum(len(ch) for ch in self.channel_map)
            print(f"  Mapped to {n_mapped} neurons across {len(self.channel_map)} channels")
