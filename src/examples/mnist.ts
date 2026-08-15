import { readFileSync } from "fs";
import NetworkController from "../core/NetworkController";
import MNISTParser from "../utils/MNISTParser";

const testSet = new MNISTParser("src/data/t10k-images.idx3-ubyte", "src/data/t10k-labels.idx1-ubyte");
const images = testSet.getImages();
const labels = testSet.getLabels();

const modelJSON = readFileSync("./src/models/mnist.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

const averageNetworkAccuracy = () => {
    let correct = 0;
    for (let i = 0; i < images.count; i++) {
        const currImage = testSet.imageAt(i);
        const expectedResult = labels[i];
        const result = network.forwardPass(currImage);
        const intResult = testSet.argMax(result);
        if (intResult === expectedResult) {
            correct++;
        }
    }

    return (correct / images.count) * 100;
}

console.log("-----------------------------------------");
console.log("MNIST Neural Network");
console.log(`Network Accuracy: ${averageNetworkAccuracy()}%`);
console.log("-----------------------------------------");