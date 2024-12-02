import tf from "@tensorflow/tfjs-node";
import sharp from "sharp";

// mapping of Fashion-MNIST labels
export const labels = [
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

const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;

export const toPixelData = async function (imgPath) {
  const pixeldata = [];
  const imageBuffer = await sharp(imgPath)
    .resize(imageWidth, imageHeight)
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { data, info } = imageBuffer;

  for (let i = 0; i < data.length; i++) {
    pixeldata.push(data[i] / 255);
  }

  return pixeldata;
};

export const runPrediction = function (model, imagepath) {
  return toPixelData(imagepath).then((pixeldata) => {
    const imageTensor = tf.tensor(pixeldata, [
      imageWidth,
      imageHeight,
      imageChannels,
    ]);
    const inputTensor = imageTensor.expandDims();
    const prediction = model.predict(inputTensor);
    const scores = prediction.arraySync()[0];

    const maxScore = prediction.max().arraySync();
    const maxScoreIndex = scores.indexOf(maxScore);

    const labelScores = {};

    scores.forEach((s, i) => {
      labelScores[labels[i]] = parseFloat(s.toFixed(4));
    });

    return {
      prediction: `${labels[maxScoreIndex]} (${parseInt(maxScore * 100)}%)`,
      scores: labelScores,
    };
  });
};

// run
const run = async function () {
  if (process.argv.length < 3) {
    console.log("please pass an image to process. ex:");
    console.log("  node test-model.js /path/to/image.jpg");
  } else {
    // e.g., /path/to/image.jpg
    const imgPath = process.argv[2];

    const modelUrl = "file://./fashion-mnist-tfjs/model.json";

    console.log("Loading model...");
    const model = await tf.loadLayersModel(modelUrl);
    model.summary();

    console.log("Running prediction...");
    const prediction = await runPrediction(model, imgPath);
    console.log(prediction);
  }
};

run();
