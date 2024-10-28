// import * as tf from "@tensorflow/tfjs";
// import * as mobilenet from "@tensorflow-models/mobilenet";

// let model;

// async function loadModel() {
//   model = await mobilenet.load();
//   console.log("MobileNet model loaded successfully!");
// }

// // Preprocess and classify the image
// async function classifyImage() {
//   const imgElement = document.getElementById("uploadedImage");

//   // Use MobileNet to classify the image
//   const predictions = await model.classify(imgElement);

//   // Display the top prediction
//   const topPrediction = predictions[0];
//   document.getElementById("result").innerText = `Prediction: ${
//     topPrediction.className
//   }, Probability: ${topPrediction.probability.toFixed(2)}`;
// }

// // Handle image upload and display
// document.getElementById("imageUpload").addEventListener("change", (event) => {
//   const file = event.target.files[0];
//   const imgElement = document.getElementById("uploadedImage");
//   imgElement.src = URL.createObjectURL(file);
//   imgElement.style.display = "block";
// });

// // Load the model when the page loads
// window.onload = loadModel;

import * as tf from "./node_modules/@tensorflow/tfjs";

// Sample code to test TensorFlow.js
(async () => {
  const tensor = tf.tensor([1, 2, 3, 4]);
  tensor.print(); // This should log a tensor with values [1, 2, 3, 4]
})();
