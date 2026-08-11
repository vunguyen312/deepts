import { readFileSync } from "fs";
import NetworkController from "../NetworkController";

// An example of a small 3-layer neural network for solving the XOR problem

const modelJSON = readFileSync("./src/models/xor.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));