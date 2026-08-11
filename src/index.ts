import { readFileSync } from "fs";
import NetworkController from "./NetworkController";
import { Layer } from "./neuralNetwork";

const modelJSON = readFileSync("./src/weights/weights.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));