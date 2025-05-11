import FFT from 'fft.js';

// Helper: Convert frequency (Hz) to mel
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

// Helper: Convert mel to frequency (Hz)
function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

// Create mel filterbank
export function createMelFilterbank(
  sampleRate: number,
  nFft: number,
  nMels: number,
  fMin: number,
  fMax: number
): Float32Array[] {
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = Array.from({ length: nMels + 2 }, (_, i) =>
    melToHz(melMin + (i * (melMax - melMin)) / (nMels + 1))
  );
  const bin = melPoints.map(f => Math.floor((nFft + 1) * f / sampleRate));
  const filterbank: Float32Array[] = Array.from({ length: nMels }, () => new Float32Array(nFft / 2 + 1).fill(0));

  for (let m = 1; m <= nMels; m++) {
    for (let k = bin[m - 1]; k < bin[m]; k++) {
      if (k >= 0 && k < filterbank[m - 1].length && bin[m] !== bin[m - 1]) {
        filterbank[m - 1][k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
      }
    }
    for (let k = bin[m]; k < bin[m + 1]; k++) {
      if (k >= 0 && k < filterbank[m - 1].length && bin[m + 1] !== bin[m]) {
        filterbank[m - 1][k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
      }
    }
  }
  return filterbank;
}

// Compute STFT (magnitude spectrogram)
export function stft(
  signal: Float32Array,
  nFft: number,
  hopLength: number
): Float32Array[] {
  const fft = new FFT(nFft);
  const frames: Float32Array[] = [];
  for (let i = 0; i + nFft <= signal.length; i += hopLength) {
    const frame = signal.slice(i, i + nFft);
    // Optionally apply a window function here (e.g., Hann)
    const out = fft.createComplexArray();
    fft.realTransform(out, frame);
    // Compute magnitude
    const mag = new Float32Array(nFft / 2 + 1);
    for (let j = 0; j < nFft / 2 + 1; j++) {
      const re = out[2 * j];
      const im = out[2 * j + 1];
      mag[j] = Math.sqrt(re * re + im * im);
    }
    frames.push(mag);
  }
  return frames;
}

export interface MelSpectrogramOptions {
  nFft?: number;
  hopLength?: number;
  nMels?: number;
  fMin?: number;
  fMax?: number;
}

// Main: Compute mel spectrogram
export function melSpectrogram(
  signal: Float32Array,
  sampleRate: number,
  options: MelSpectrogramOptions = {}
): Float32Array[] {
  const {
    nFft = 2048,
    hopLength = 512,
    nMels = 128,
    fMin = 0,
    fMax = sampleRate / 2,
  } = options;

  // Pad signal by nFft // 2 on both sides (to match torchaudio)
  const pad = Math.floor(nFft / 2);
  const padded = new Float32Array(signal.length + 2 * pad);
  padded.set(signal, pad);

  const spec = stft(padded, nFft, hopLength); // [frames][freq]
  const filterbank = createMelFilterbank(sampleRate, nFft, nMels, fMin, fMax);
  // Apply mel filterbank
  const melSpec = spec.map(frame => {
    const melFrame = new Float32Array(nMels);
    for (let m = 0; m < nMels; m++) {
      for (let k = 0; k < frame.length; k++) {
        melFrame[m] += frame[k] * filterbank[m][k];
      }
    }
    return melFrame;
  });
  return melSpec; // [frames][nMels]
}