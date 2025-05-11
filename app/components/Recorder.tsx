import React, { useEffect, useRef, useState } from 'react';
import { melSpectrogram } from '../lib/mel'; // adjust path as needed
import { loadOnnxModel, runOnnxInference } from '../lib/onnx'; // add this import

const SAMPLE_RATE = 44100; // or 44100, but match your model
const WINDOW_SECONDS = 5;
const BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS;
const MODEL_URL = '/tiny_mel_classifier.onnx'; // adjust path as needed
const ONNX_INPUT_NAME = 'mel'; // change if your model uses a different input name

const Recorder: React.FC = () => {
  const audioBufferRef = useRef<Float32Array>(new Float32Array(BUFFER_SIZE));
  const bufferOffsetRef = useRef(0);
  const [classLabel, setClassLabel] = useState<Number>(-1);

  useEffect(() => {
    let audioContext: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;

    // Load ONNX model once
    loadOnnxModel(MODEL_URL);

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      console.log('Actual audio context sample rate:', audioContext.sampleRate);
      source = audioContext.createMediaStreamSource(stream);

      // Use 4096 for compatibility, but you can try 2048 or 8192
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = async (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const buffer = audioBufferRef.current;
        let offset = bufferOffsetRef.current;

        // Copy input to our buffer
        for (let i = 0; i < input.length && offset < BUFFER_SIZE; i++, offset++) {
          buffer[offset] = input[i];
        }
        bufferOffsetRef.current = offset;

        // If buffer is full (5s), process it
        if (offset >= BUFFER_SIZE) {
          const chunk = new Float32Array(buffer);

          const mel = melSpectrogram(chunk, SAMPLE_RATE, {
            nFft: 512,
            hopLength: 256,
            nMels: 64,
            fMin: 0,
            fMax: SAMPLE_RATE / 2,
          });
          console.log('Mel spectrogram shape:', mel.length, 'frames ×', mel[0]?.length, 'nMels');

          // mel: [frames][nMels] => transpose to [nMels][frames]
          const nMels = 64;
          const seqLen = mel.length;
          const melTransposed = Array.from({ length: nMels }, (_, m) =>
            mel.map(frame => frame[m])
          );

          // Flatten to Float32Array in (1, 64, seqLen) order
          const inputArray = new Float32Array(nMels * seqLen);
          for (let m = 0; m < nMels; m++) {
            for (let t = 0; t < seqLen; t++) {
              inputArray[m * seqLen + t] = melTransposed[m][t];
            }
          }

          // Run ONNX inference
          try {
            const start = performance.now();
            const results = await runOnnxInference(ONNX_INPUT_NAME, inputArray, [1, 1, nMels, seqLen]);
            const argmax = results.output.data.indexOf(Math.max(...results.output.data));
            const end = performance.now();
            console.log('ONNX results:', results, 'Argmax:', argmax, 'Time:', end - start);
            setClassLabel(argmax);
          } catch (err) {
            console.error('ONNX inference error:', err);
          }

          bufferOffsetRef.current = 0;
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    });

    return () => {
      processor?.disconnect();
      source?.disconnect();
      audioContext?.close();
    };
  }, []);

  return (
    <div>
      {classLabel === -1 ? (
        <div>Recording and processing 5s windows...</div>
      ) : (
        <div>Class label: {classLabel}</div>
      )}
    </div>
  );
};

export default Recorder;