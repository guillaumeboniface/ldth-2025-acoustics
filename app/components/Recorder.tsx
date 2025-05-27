import React, { useEffect, useRef, useState } from 'react';
import { mel_spectrogram_db } from "rust-melspec-wasm";
import { loadOnnxModel, runOnnxInference } from '../lib/onnx'; // add this import

const SAMPLE_RATE = 44100; // or 44100, but match your model
const WINDOW_SECONDS = 5;
const BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS;
const MODEL_URL = '/tiny_mel_classifier.onnx'; // adjust path as needed
const ONNX_INPUT_NAME = 'mel'; // change if your model uses a different input name
const EXPECTED_SEQ_LEN = 862; // Expected sequence length by the model
const INFERENCE_INTERVAL_MS = 100; // Run inference every 0.1 seconds

const Recorder: React.FC = () => {
  const audioBufferRef = useRef<Float32Array>(new Float32Array(BUFFER_SIZE));
  const bufferOffsetRef = useRef(0);
  const isBufferFullRef = useRef(false);
  const [classLabel, setClassLabel] = useState<Number>(-1);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    let audioContext: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    let inferenceInterval: NodeJS.Timeout | null = null;

    // Load ONNX model once
    loadOnnxModel(MODEL_URL);

    const runInference = async () => {
      if (!isBufferFullRef.current || isProcessing) return;
      
      setIsProcessing(true);
      try {
        // Get current 5-second window from circular buffer
        const buffer = audioBufferRef.current;
        const chunk = new Float32Array(BUFFER_SIZE);
        
        // Copy the circular buffer to a linear array
        const offset = bufferOffsetRef.current;
        if (offset === 0) {
          // Buffer hasn't wrapped, use as is
          chunk.set(buffer);
        } else {
          // Buffer has wrapped, reconstruct the correct order
          chunk.set(buffer.slice(offset), 0);
          chunk.set(buffer.slice(0, offset), BUFFER_SIZE - offset);
        }

        const mel = mel_spectrogram_db(SAMPLE_RATE, chunk, 512, 512, 256, 0, SAMPLE_RATE / 2, 64, 120);
        
        // Pad or truncate to expected sequence length
        let paddedMel = mel;
        if (mel.length < EXPECTED_SEQ_LEN) {
          const lastFrame = mel[mel.length - 1];
          const paddingFrames = EXPECTED_SEQ_LEN - mel.length;
          paddedMel = [...mel, ...Array(paddingFrames).fill(lastFrame)];
        } else if (mel.length > EXPECTED_SEQ_LEN) {
          paddedMel = mel.slice(0, EXPECTED_SEQ_LEN);
        }

        // mel: [frames][nMels] => transpose to [nMels][frames]
        const nMels = 64;
        const seqLen = paddedMel.length;
        const melTransposed = Array.from({ length: nMels }, (_, m) =>
          paddedMel.map(frame => frame[m])
        );

        // Flatten to Float32Array in (1, 64, seqLen) order
        const inputArray = new Float32Array(nMels * seqLen);
        for (let m = 0; m < nMels; m++) {
          for (let t = 0; t < seqLen; t++) {
            inputArray[m * seqLen + t] = melTransposed[m][t];
          }
        }

        // Run ONNX inference
        const start = performance.now();
        const results = await runOnnxInference(ONNX_INPUT_NAME, inputArray, [1, 1, nMels, seqLen]);
        const outputData = Array.from(results.output.data as Float32Array);
        const argmax = outputData.indexOf(Math.max(...outputData));
        const end = performance.now();
        console.log('ONNX results:', results, 'Argmax:', argmax, 'Time:', end - start);
        setClassLabel(argmax);
      } catch (err) {
        console.error('ONNX inference error:', err);
      } finally {
        setIsProcessing(false);
      }
    };

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      console.log('Actual audio context sample rate:', audioContext.sampleRate);
      source = audioContext.createMediaStreamSource(stream);

      // Use 4096 for compatibility, but you can try 2048 or 8192
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const buffer = audioBufferRef.current;
        let offset = bufferOffsetRef.current;

        // Copy input to circular buffer
        for (let i = 0; i < input.length; i++) {
          buffer[offset] = input[i];
          offset = (offset + 1) % BUFFER_SIZE;
        }
        bufferOffsetRef.current = offset;

        // Mark buffer as full once we've collected 5 seconds
        if (!isBufferFullRef.current && offset >= BUFFER_SIZE - input.length) {
          isBufferFullRef.current = true;
          console.log('Buffer full, starting inference timer');
        }
      };

      // Start inference timer once we have enough data
      inferenceInterval = setInterval(runInference, INFERENCE_INTERVAL_MS);

      source.connect(processor);
      processor.connect(audioContext.destination);
    });

    return () => {
      if (inferenceInterval) clearInterval(inferenceInterval);
      processor?.disconnect();
      source?.disconnect();
      audioContext?.close();
    };
  }, []);

  return (
    <div 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: classLabel === 1 ? '#ff0000' : '#00ff00',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '24px',
        fontWeight: 'bold',
        color: 'white',
        textShadow: '2px 2px 4px rgba(0,0,0,0.5)',
        zIndex: 9999
      }}
    >
      {classLabel === -1 ? (
        <div>Recording and processing 5s windows...</div>
      ) : (
        <div>
          Class label: {classLabel}
          {isProcessing && <div style={{ fontSize: '16px', marginTop: '10px' }}>Processing...</div>}
        </div>
      )}
    </div>
  );
};

export default Recorder;