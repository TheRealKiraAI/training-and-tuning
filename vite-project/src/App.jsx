import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import classifyMoonPhase from "./moonClassifier";
import trainModel from "./trainModel";

import * as mobilenet from "@tensorflow-models/mobilenet";
import "@tensorflow/tfjs";

const App = () => {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [result, setResult] = useState("");

  const [text, setText] = useState("Awaiting MobileNet Model...");

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await mobilenet.load();
      setModel(loadedModel);
      setText("MobileNet model loaded successfully!");
      console.log("MobileNet model loaded successfully!");
    };
    loadModel();
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImage(URL.createObjectURL(file));
  };

  // const classifyImage = async () => {
  //   if (image) {
  //     const prediction = await classifyMoon(image);
  //     console.log("classifying...");
  //     setResult(prediction);
  //   }
  // };

  const classifyImage = async () => {
    if (!model || !image) return;

    const imgElement = document.getElementById("uploadedImage");
    const predictions = await model.classify(imgElement);

    const topPrediction = predictions[0];
    setPrediction(
      `Prediction: ${
        topPrediction.className
      }, Probability: ${topPrediction.probability.toFixed(2)}`
    );
  };

  return (
    <div>
      <h1>Image Classifier with MobileNet</h1>
      <input type="file" onChange={handleImageUpload} />

      {text}

      {image && (
        <img
          id="uploadedImage"
          src={image}
          alt="Uploaded"
          style={{ display: "block", maxWidth: "300px", margin: "20px 0" }}
          onLoad={classifyImage}
        />
      )}

      {prediction && <h2 id="result">Prediction: {prediction}</h2>}
      <button onClick={trainModel}>Train Model</button>
    </div>
  );
};

export default App;
