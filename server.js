var tf = require('@tensorflow/tfjs');
const readline = require('readline');
require('@tensorflow/tfjs-node')
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
const csv = require('csv-parser');
const fs = require('fs');
Canvas = require('canvas');
const IMG_Width = 224;
const IMG_Height = 224;
const IMAGE_SIZE = IMG_Height * IMG_Width;

const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.

// Name prefixes of layers that will be unfrozen during fine-tuning.
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11'];

// Name of the layer that will become the top layer of the truncated base.
const topLayerName =
  `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;
// Used to scale the first column (0-1 shape indicator) of `yTrue`
// in order to ensure balanced contributions to the final loss value
// from shape and bounding-box predictions.
const LABEL_MULTIPLIER = [CANVAS_SIZE, 1, 1, 1, 1];


var NUM_CLASSES;
var CLASSES;
var currentModel;
var datas;
var xs;
var ys;
mainFuction();


/**
 * Custom loss function for object detection.
 *
 * The loss function is a sum of two losses
 * - shape-class loss, computed as binaryCrossentropy and scaled by
 *   `classLossMultiplier` to match the scale of the bounding-box loss
 *   approximatey.
 * - bounding-box loss, computed as the meanSquaredError between the
 *   true and predicted bounding boxes.
 * @param {tf.Tensor} yTrue True labels. Shape: [batchSize, 5].
 *   The first column is a 0-1 indicator for whether the shape is a triangle
 *   (0) or a rectangle (1). The remaining for columns are the bounding boxes
 *   for the target shape: [left, right, top, bottom], in unit of pixels.
 *   The bounding box values are in the range [0, CANVAS_SIZE).
 * @param {tf.Tensor} yPred Predicted labels. Shape: the same as `yTrue`.
 * @return {tf.Tensor} Loss scalar.
 */
function customLossFunction(yTrue, yPred) {
  return tf.tidy(() => {
    // Scale the the first column (0-1 shape indicator) of `yTrue` in order
    // to ensure balanced contributions to the final loss value
    // from shape and bounding-box predictions.
    return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred);
  });
}

/**
 * Loads MobileNet, removes the top part, and freeze all the layers.
 *
 * The top removal and layer freezing are preparation for transfer learning.
 *
 * Also gets handles to the layers that will be unfrozen during the fine-tuning
 * phase of the training.
 *
 * @return {tf.Model} The truncated MobileNet, with all layers frozen.
 */
async function loadTruncatedBase() {
  // TODO(cais): Add unit test.
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const fineTuningLayers = [];
  const layer = mobilenet.getLayer(topLayerName);
  const truncatedBase =
    tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  // Freeze the model's layers.
  for (const layer of truncatedBase.layers) {
    layer.trainable = false;
    for (const groupName of topLayerGroupNames) {
      if (layer.name.indexOf(groupName) === 0) {
        fineTuningLayers.push(layer);
        break;
      }
    }
  }

  // tf.util.assert(
  //   fineTuningLayers.length > 1,
  //   `Did not find any layers that match the prefixes ${topLayerGroupNames}`);
  return { truncatedBase, fineTuningLayers };
}


/**
 * Build a new head (i.e., output sub-model) that will be connected to
 * the top of the truncated base for object detection.
 *
 * @param {tf.Shape} inputShape Input shape of the new model.
 * @returns {tf.Model} The new head model.
 */
function buildNewHead(inputShape) {
  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten({ inputShape }));
  newHead.add(tf.layers.dense({ units: 200, activation: 'relu' }));
  // Five output units:
  //   - The first is a shape indictor: predicts whether the target
  //     shape is a triangle or a rectangle.
  //   - The remaining four units are for bounding-box prediction:
  //     [left, right, top, bottom] in the unit of pixels.
  newHead.add(tf.layers.dense({ units: 5 }));
  return newHead;
}

/**
 * Builds object-detection model from MobileNet.
 *
 * @returns {[tf.Model, tf.layers.Layer[]]}
 *   1. The newly-built model for simple object detection.
 *   2. The layers that can be unfrozen during fine-tuning.
 */
async function buildObjectDetectionModel() {
  const { truncatedBase, fineTuningLayers } = await loadTruncatedBase();

  // Build the new head model.
  const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1));
  const newOutput = newHead.apply(truncatedBase.outputs[0]);
  const model = tf.model({ inputs: truncatedBase.inputs, outputs: newOutput });

  return { model, fineTuningLayers };
}
async function mainFuction() {
  const { model, fineTuningLayers } = await buildObjectDetectionModel();
  model.compile({ loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3) });
  console.log('Phase 1...');
  model.summary();
  // // Initial phase of transfer learning.
  // await model.fit(images, targets, {
  //   epochs: 50,
  //   batchSize: 2,
  //   callbacks: {
  //     onBatchEnd: async (batch, logs) => {
  //       console.log('Loss: ' + logs.loss.toFixed(5));
  //       console.log('accuracy: ' + logs.acc);
  //     }
  //   }
  // });
  // // Fine-tuning phase of transfer learning.
  // // Unfreeze layers for fine-tuning.
  // for (const layer of fineTuningLayers) {
  //   layer.trainable = true;
  // }
  // model.compile({ loss: customLossFunction, optimizer: tf.train.rmsprop(2e-3) });
  // model.summary();

  // // Do fine-tuning.
  // // The batch size is reduced to avoid CPU/GPU OOM. This has
  // // to do with the unfreezing of the fine-tuning layers above,
  // // which leads to higher memory consumption during backpropagation.
  // console.log('Phase 2 of 2: fine-tuning phase');
  // await model.fit(images, targets, {
  //   epochs: 50,
  //   batchSize: args.batchSize / 2,
  //   callbacks: {
  //     onBatchEnd: async (batch, logs) => {
  //       console.log('Loss: ' + logs.loss.toFixed(5));
  //       console.log('accuracy: ' + logs.acc);
  //     }
  //   }
  // });
}












  // readFileCsv();

  // function readFileCsv() {
  //   const results = [];
  //   const labels = [];
  //   fs.createReadStream('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train.csv')
  //     .pipe(csv())
  //     .on('data', data => {
  //       results.push(data);
  //       labels.push(data.label);
  //     })
  //     .on('end', async () => {
  //       datas = results;
  //       CLASSES = labels.filter(distinct);
  //       NUM_CLASSES = CLASSES.length;
  //       truncatedMobileNet = await loadTruncatedMobileNet();
  //       truncatedMobileNet.summary()
  //       try {
  //         currentModel = await tf.loadLayersModel('file://D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/model.json');
  //       } catch (error) {
  //         currentModel = createConvModel();
  //         // await currentModel.save('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/');
  //         getTrainData();
  //         console.log('start train');
  //         await train(currentModel);
  //         console.log('train done');
  //         currentModel.save('file://D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition');
  //         // const pre = currentModel.predict(getImgAndResize('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/0b4c2cf7352f40a3b55339f13c04bcda.png'))
  //         // const arr = pre.arraySync();
  //         // console.log('arr', arr);
  //         // console.log('max score', Math.max(...arr[0]));
  //         // const index = arr[0].indexOf(Math.max(...arr[0]))
  //         // console.log('max score', CLASSES[index]);
  //       }
  //       // input();
  //       predictFolder();
  //     });
  // }

  // function input() {
  //   rl.question('Input path image to predict: ', (answer) => {
  //     if (answer !== 'stop') {
  //       predict(answer);
  //     } else {
  //       rl.close();
  //     }
  //   });
  // }
  // function moveImage2() {
  //   fs.readdir('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/', function (err, files) {
  //     //handling error
  //     if (err) {
  //       return console.log('Unable to scan directory: ' + err);
  //     }
  //     //listing all files using forEach
  //     files.forEach(function (file) {
  //       // Do whatever you want to do with the file
  //       var img = new Canvas.Image(); // Create a new Image
  //       img.src = 'D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/' + file;
  //       var canvas = Canvas.createCanvas(img.width, img.height);
  //       var ctx = canvas.getContext('2d');
  //       ctx.drawImage(img, 0, 0);
  //       let pixels = ctx.getImageData(0, 0, img.width, img.height);
  //       for (let y = 0; y < pixels.height; y++) {
  //         for (let x = 0; x < pixels.width; x++) {
  //           let i = (y * 4) * pixels.width + x * 4;
  //           let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;
  //           pixels.data[i] = avg;
  //           pixels.data[i + 1] = avg;
  //           pixels.data[i + 2] = avg;
  //         }
  //       }
  //       ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);
  //       var buf = canvas.toBuffer();
  //       fs.writeFileSync('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test_move/' + file, buf);
  //     });
  //   });

  // }

  // function moveImage(path, label) {
  //   var img = new Canvas.Image(); // Create a new Image
  //   img.src = 'D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/' + path;
  //   var canvas = Canvas.createCanvas(img.width, img.height);
  //   var ctx = canvas.getContext('2d');
  //   ctx.drawImage(img, 0, 0);
  //   let pixels = ctx.getImageData(0, 0, img.width, img.height);
  //   for (let y = 0; y < pixels.height; y++) {
  //     for (let x = 0; x < pixels.width; x++) {
  //       let i = (y * 4) * pixels.width + x * 4;
  //       let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;
  //       pixels.data[i] = avg;
  //       pixels.data[i + 1] = avg;
  //       pixels.data[i + 2] = avg;
  //     }
  //   }
  //   ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);
  //   var buf = canvas.toBuffer();
  //   fs.writeFileSync('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train_move/' + 'Label' + label + '_' + path, buf);
  // }
  // function logFile(content) {
  //   fs.writeFile('log.txt', content, (err) => {
  //     if (err) console.log(err);
  //     console.log('Successfully Written to File.');
  //   });
  // }

  // function predict(path) {
  //   const pre = currentModel.predict(truncatedMobileNet.predict(getImgAndResize('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/' + path)))
  //   const arr = pre.arraySync();
  //   const maxScore = Math.max(...arr[0]);
  //   const index = arr[0].indexOf(maxScore)
  //   return { maxScore: maxScore, class: CLASSES[index] };
  // }
  // function predictFolder() {
  //   let i = 0;
  //   fs.readdir('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/', function (err, files) {
  //     if (err) {
  //       return console.log('Unable to scan directory: ' + err);
  //     }
  //     let lines = '';
  //     for (const file of files) {
  //       if (i++ > 1000) {
  //         break;
  //       }
  //       const pre = predict(file);
  //       if (pre.maxScore >= 0.5) {
  //         lines += file + ' : ' + pre.class + ' : ' + pre.maxScore + '\n';
  //       }
  //     }
  //     logFile(lines);
  //   });
  // }
  // async function loadTruncatedMobileNet() {
  //   const mobilenet = await tf.loadLayersModel(
  //     'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  //   const layer = mobilenet.getLayer('conv_pw_13_relu');
  //   return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  // }

  // function createConvModel() {
  //   model = tf.sequential({
  //     layers: [
  //       tf.layers.flatten(
  //         { inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),
  //       tf.layers.dense({
  //         units: NUM_CLASSES,
  //         activation: 'relu',
  //         kernelInitializer: 'varianceScaling',
  //         useBias: true
  //       }),
  //       tf.layers.dense({
  //         units: NUM_CLASSES,
  //         kernelInitializer: 'varianceScaling',
  //         useBias: false,
  //         activation: 'softmax'
  //       })
  //     ]
  //   });
  //   return model;
  // }

  // function getTrainData() {
  //   var i = 0;
  //   for (const item of datas) {
  //     i++;
  //     const y = tf.tidy(
  //       () => tf.oneHot(tf.tensor1d([+item.label]).toInt(), NUM_CLASSES));
  //     if (!xs) {
  //       xs = tf.keep(truncatedMobileNet.predict(getImgAndResize(
  //         'D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/' + item.image
  //       )));
  //       ys = tf.keep(y);
  //     } else {
  //       const oldX = xs;
  //       xs = tf.keep(
  //         oldX.concat(
  //           truncatedMobileNet.predict(getImgAndResize(
  //             'D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/' + item.image
  //           )),
  //           0
  //         )
  //       );
  //       const oldY = ys;
  //       ys = tf.keep(oldY.concat(y, 0));
  //       oldX.dispose();
  //       oldY.dispose();
  //       y.dispose();
  //     }
  //     console.log('i:' + i);
  //   }
  //   ys.print();
  // }
  // function getImgAndResize(path) {
  //   var img = new Canvas.Image(); // Create a new Image
  //   img.src = path;
  //   var canvas = Canvas.createCanvas(img.width, img.height);
  //   var ctx = canvas.getContext('2d');
  //   ctx.drawImage(img, 0, 0);
  //   var image = tf.browser.fromPixels(canvas);
  //   image = tf.image.resizeBilinear(image, [IMG_Height, IMG_Width], false);
  //   const batchedImage = image.expandDims(0);
  //   return batchedImage
  //     .toFloat()
  //     .div(tf.scalar(127))
  //     .sub(tf.scalar(1));
  // }

  // function getImgResizeAndGrayScale(path) {
  //   var img = new Canvas.Image(); // Create a new Image
  //   img.src = path;
  //   var canvas = Canvas.createCanvas(img.width, img.height);
  //   var ctx = canvas.getContext('2d');
  //   ctx.drawImage(img, 0, 0);
  //   let pixels = ctx.getImageData(0, 0, img.width, img.height);
  //   for (let y = 0; y < pixels.height; y++) {
  //     for (let x = 0; x < pixels.width; x++) {
  //       let i = (y * 4) * pixels.width + x * 4;
  //       let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;
  //       pixels.data[i] = avg;
  //       pixels.data[i + 1] = avg;
  //       pixels.data[i + 2] = avg;
  //     }
  //   }
  //   ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);

  //   var image = tf.browser.fromPixels(canvas, 1);
  //   image = tf.image.resizeBilinear(image, [IMG_Height, IMG_Width], false);
  //   const batchedImage = image.expandDims(0);
  //   return batchedImage
  //     .toFloat()
  //     .div(tf.scalar(127))
  //     .sub(tf.scalar(1));
  // }

  // const distinct = (value, index, self) => {
  //   return self.indexOf(value) === index;
  // };

  // async function train(model) {
  //   const batchSize = 200;
  //   const optimizer = tf.train.adam(0.0001);
  //   model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  //   await model.fit(xs, ys, {
  //     batchSize,
  //     epochs: 60,
  //     callbacks: {
  //       onBatchEnd: async (batch, logs) => {
  //         console.log('Loss: ' + logs.loss.toFixed(5));
  //         console.log('accuracy: ' + logs.acc);
  //       }
  //     }
  //   }).then(info => {
  //     console.log('Final accuracy', info.history.acc);
  //   });
  // }

