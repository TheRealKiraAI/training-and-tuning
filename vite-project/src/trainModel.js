import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";

const NUM_CLASSES = 4; // number of moon phases; also number of classifications in directories of images

let model = tf.sequential();
model.add(
  tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
);
model.add(
  tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
);

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: "adam",
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss:
    CLASS_NAMES.length === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ["accuracy"],
});

const trainModel = async () => {
  // 1. Prepare dataset to do

  // load pre-trained MobileNet image classification model
  const mobilenetBaseModel = await mobilenet.load();
  console.log("mobilenetBaseModel", mobilenetBaseModel);

  // freeze base layers to retain learned features
  if (mobilenetBaseModel && mobilenetBaseModel.layers) {
    mobilenetBaseModel.layers.forEach((layer) => {
      layer.trainable = false;
    });
    console.log("mobilenet freeze layers");
  } else {
    console.error("mobilenetBaseModel undefined...........");
  }

  // create new model on top of pre-trained model
  const model = tf.sequential();
  if (mobilenetBaseModel && mobilenetBaseModel.layers) {
    model.add(mobilenetBaseModel);
  } else {
    console.error("mobilenetBaseModel not defined...........");
  }

  // add custom layers for moon phase classification
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  console.log("mobilenet add layers");

  model.summary();

  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("mobilenet compile");

  // // Prepare your dataset (this is a placeholder, implement your own data loading)
  // const { xs, ys } = await loadData(); // Implement loadData to load your images and labels

  // // Train the model
  // await model.fit(xs, ys, {
  //   epochs: 10, // Adjust the number of epochs as needed
  //   batchSize: 32, // Adjust the batch size as needed
  //   validationSplit: 0.2, // Use 20% of the data for validation
  // });

  // // Save the model
  // await model.save('localstorage://my-custom-mobilenet-model');
};

export default trainModel;
