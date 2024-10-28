import * as tf from "@tensorflow/tfjs";

let model;

const loadModel = async () => {
  if (!model) {
    model = await tf.loadLayersModel("/path/to/model.json"); // Use your model URL here
  }
  return model;
};

const classifyMoonPhase = async (imageData) => {
  await loadModel();

  const img = new Image();
  img.src = imageData;
  await img.decode();

  const tensor = tf.browser
    .fromPixels(img)
    .resizeNearestNeighbor([224, 224]) // Resize as per your model’s requirements
    .toFloat()
    .expandDims();

  const predictions = model.predict(tensor);
  const moonPhase = predictions.argMax(-1).dataSync()[0];

  return moonPhase === 0
    ? "New Moon"
    : moonPhase === 1
    ? "First Quarter"
    : moonPhase === 2
    ? "Full Moon"
    : "Last Quarter"; // Example mapping
};

export default classifyMoonPhase;
