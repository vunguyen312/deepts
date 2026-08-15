import { readFileSync } from "fs";
import NetworkController from "../core/NetworkController";

// An example of a small 3-layer neural network for solving the XOR problem

const modelJSON = readFileSync("./src/models/xor.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forwardPass([1, 0]));
console.log("Result of [0, 0]: " + network.forwardPass([0, 0]));
console.log("Result of [1, 1]: " + network.forwardPass([1, 1]));
console.log("Result of [0, 1]: " + network.forwardPass([0, 1]));
console.log("-----------------------------------------");