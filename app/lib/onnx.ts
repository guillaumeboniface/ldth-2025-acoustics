import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

/**
 * Loads an ONNX model from the given URL or path.
 * Only loads once; subsequent calls are no-ops.
 */
export async function loadOnnxModel(modelUrl: string) {
  if (!session) {
    session = await ort.InferenceSession.create(modelUrl);
  }
  return session;
}

/**
 * Runs inference on the loaded ONNX model.
 * @param inputName The name of the model's input tensor.
 * @param inputData The input data as a Float32Array or appropriate TypedArray.
 * @param inputShape The shape of the input tensor.
 * @returns The model's output(s).
 */
export async function runOnnxInference(
  inputName: string,
  inputData: Float32Array,
  inputShape: number[]
) {
  if (!session) {
    throw new Error('ONNX model not loaded. Call loadOnnxModel() first.');
  }
  const tensor = new ort.Tensor('float32', inputData, inputShape);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[inputName] = tensor;
  const results = await session.run(feeds);
  return results;
}