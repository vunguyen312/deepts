import { join } from "node:path";
import { createNetwork, freezeToJSON, LinearLayer, SGD, MNISTParser, Sigmoid, 
         ReLU } from "../src/index";

const BATCH_SIZE = 64;
const NUM_EPOCHS = 30;
const LEARNING_RATE = 0.00625;
const MOMENTUM = 0.9;

const network = createNetwork(
    [
        new LinearLayer(784, 30),
        new ReLU(),
        new LinearLayer(30, 10),
        new Sigmoid()
    ]
);
const optimizer = new SGD(network.params, LEARNING_RATE, MOMENTUM);

const trainingSet = new MNISTParser(
    join(__dirname, "../data/train-images.idx3-ubyte"),
    join(__dirname, "../data/train-labels.idx1-ubyte")
);
const trainingImages = trainingSet.getImages();
const trainingLabels = trainingSet.getLabels();

const before = new Date().getTime();
for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    for (let start = 0; start < trainingImages.count; start += BATCH_SIZE) {
        const end = Math.min(start + BATCH_SIZE, trainingImages.count);

        optimizer.zeroGrad();
        for (let j = start; j < end; j++) {
            network.backward(
                trainingSet.imageAt(j),
                trainingSet.oneHot(trainingLabels[j])
            );
        }
        optimizer.step();
    }
}
const after = new Date().getTime();
console.log("Time to train is: " + (after - before) / 60000 + "mins");

const testSet = new MNISTParser(
    join(__dirname, "../data/t10k-images.idx3-ubyte"),
    join(__dirname, "../data/t10k-labels.idx1-ubyte")
);
const images = testSet.getImages();
const labels = testSet.getLabels();

const averageNetworkAccuracy = () => {
    let correct = 0;
    for (let i = 0; i < images.count; i++) {
        const currImage = testSet.imageAt(i);
        const expectedResult = labels[i];
        const result = network.forward(currImage);
        const intResult = testSet.argMax(result.data);
        if (intResult === expectedResult) {
            correct++;
        }
    }

    return (correct / images.count) * 100;
};

console.log("-----------------------------------------");
console.log("MNIST Neural Network");
console.log(`Network Accuracy: ${averageNetworkAccuracy().toFixed(2)}%`);
console.log("-----------------------------------------");

freezeToJSON(network, join(__dirname, "./weights/mnist2.json"));