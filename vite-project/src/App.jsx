import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import classifyMoonPhase from "./moonClassifier";

const App = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleClassify = async () => {
    if (image) {
      const prediction = await classifyMoonPhase(image);
      setResult(prediction);
    }
  };

  return (
    <div>
      <h1>Moon Phase Classifier</h1>
      <input type="file" onChange={handleImageUpload} />
      <button onClick={handleClassify}>Classify</button>
      {image && <img src={image} alt="Uploaded" style={{ width: "200px" }} />}
      {result && <h2>Prediction: {result}</h2>}
    </div>
  );
};

export default App;
