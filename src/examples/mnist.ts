import { readFileSync } from "node:fs";
import { join } from "node:path";
import NetworkController from "../core/NetworkController";
import MNISTParser from "../utils/MNISTParser";

// Requires the dataset: run `npm run data` (or scripts/download-mnist.sh) first.

const testSet = new MNISTParser(
    join(__dirname, "../data/t10k-images.idx3-ubyte"),
    join(__dirname, "../data/t10k-labels.idx1-ubyte")
);
const images = testSet.getImages();
const labels = testSet.getLabels();

const modelPath = join(__dirname, "../weights/mnist.json");
const modelJSON = readFileSync(modelPath, "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

const averageNetworkAccuracy = () => {
    let correct = 0;
    for (let i = 0; i < images.count; i++) {
        const currImage = testSet.imageAt(i);
        const expectedResult = labels[i];
        const result = network.forward(currImage);
        const intResult = testSet.argMax(result);
        if (intResult === expectedResult) {
            correct++;
        }
    }

    return (correct / images.count) * 100;
};

console.log("-----------------------------------------");
console.log("MNIST Neural Network");
console.log(`Network Accuracy: ${averageNetworkAccuracy()}%`);
console.log("-----------------------------------------");
