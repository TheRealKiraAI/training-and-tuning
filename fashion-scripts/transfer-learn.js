import tf from "@tensorflow/tfjs-node";

// Fashion-MNIST training & test data
const trainDataUrl = "file://./fashion-mnist/fashion-mnist_train.csv";
const testDataUrl = "file://./fashion-mnist/fashion-mnist_test.csv";

// mapping of Fashion-MNIST labels (i.e., T-shirt=0, Trouser=1, etc.)
const labels = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

const numOfClasses = 5;

const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;

const batchSize = 100;
const epochsValue = 5;

// load and transform data
const loadData = function (dataUrl, batches = batchSize) {
  // normalize data values between 0-1
  // normalizes pixel values from 0 - 255 to 0 - 1
  const normalize = ({ xs, ys }) => {
    return {
      xs: Object.values(xs).map((x) => x / 255),
      ys: ys.label,
    };
  };

  // transform input array (xs) to 3D tensor
  // binarize output label (ys)
  const transform = ({ xs, ys }) => {
    // array of zeros
    const zeros = new Array(numOfClasses).fill(0);

    return {
      xs: tf.tensor(xs, [imageWidth, imageHeight, imageChannels]),
      ys: tf.tensor1d(
        zeros.map((z, i) => {
          return i === ys - numOfClasses ? 1 : 0; // changing data because same csv, not normally
        })
      ),
    };
  };

  // load, normalize, transform, batch
  return tf.data
    .csv(dataUrl, { columnConfigs: { label: { isLabel: true } } })
    .map(normalize)
    .filter((f) => f.ys >= labels.length - numOfClasses)
    .map(transform)
    .batch(batchSize); // batches it all together
};

// Define the model architecture
const buildModel = function (baseModel) {
  // remove last layer of base model. This is softmax classification layer for classifying Fashion-MNIST. Leaves us with Flatten layer as new final layer.
  baseModel.layers.pop();

  // freeze weights in base model layers so they don't change when we train new model
  for (const layer of baseModel.layers) {
    layer.trainable = false;
  }

  // create new sequential model starting from the layers of previous model
  const model = tf.sequential({
    layers: baseModel.layers,
  });
  model.add(
    tf.layers.dense({
      units: numOfClasses,
      activation: "softmax", // classification model; trainable layers
      name: "topSoftMax",
    })
  );

  // configure & compile the model
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

// train the model against the training data
const trainModel = async function (model, trainingData, epochs = epochsValue) {
  const options = {
    epochs: epochs,
    verbose: 0,
    callbacks: {
      onEpochBegin: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} of ${epochs} ...`);
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(`  train-set loss: ${logs.loss.toFixed(4)}`);
        console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`);
      },
    },
  };

  return await model.fitDataset(trainingData, options);
};

// verify the model against the test data
const evaluateModel = async function (model, testingData) {
  const result = await model.evaluateDataset(testingData);
  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
  console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
};

// run
const run = async function () {
  const trainData = loadData(trainDataUrl);
  const testData = loadData(testDataUrl);

  // const arr = await trainData.take(1).toArray();
  // arr[0].ys.print();
  // arr[0].xs.print();

  const amount = Math.floor(3000 / batchSize);
  const trainDataSubset = trainData.take(amount); // 10% of data

  const baseModelUrl = "file://./fashion-mnist-tfjs/model.json";
  const saveModelPath = "file://./fashion-mnist-tfjs-transfer";

  const baseModel = await tf.loadLayersModel(baseModelUrl);
  const model = buildModel(baseModel);
  model.summary();

  const info = await trainModel(model, trainData);
  console.log("\r\n", info);
  console.log("\r\nEvaluating model...");
  await evaluateModel(model, testData);
  console.log("\r\nSaving model...");
  await model.save(saveModelPath);
};

run();
