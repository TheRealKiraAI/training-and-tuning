import * as tf from "@tensorflow/tfjs";

let model;

// Load the pre-trained model from the `model` directory
async function loadModel() {
  model = await tf.loadGraphModel("model/model.json");
  console.log("Model loaded successfully!");
}

// Preprocess the image for the model
function preprocessImage(imageElement) {
  let tensor = tf.browser
    .fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224]) // Resize to the input size of the model
    .toFloat()
    .expandDims();
  return tensor;
}

// Classify the uploaded image
async function classifyImage() {
  const imgElement = document.getElementById("uploadedImage");
  const preprocessedImage = preprocessImage(imgElement);
  const predictions = await model.predict(preprocessedImage).data();

  // Find the most confident prediction and display it
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const moonPhases = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]; // Add your labels here
  document.getElementById("result").innerText = `Prediction: ${
    moonPhases[maxIndex]
  }, Confidence: ${predictions[maxIndex].toFixed(2)}`;
}

// Handle image upload and display
document.getElementById("imageUpload").addEventListener("change", (event) => {
  const file = event.target.files[0];
  const imgElement = document.getElementById("uploadedImage");
  imgElement.src = URL.createObjectURL(file);
  imgElement.style.display = "block";
});

// Load the model when the page loads
window.onload = loadModel;
