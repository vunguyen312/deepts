import { readFileSync } from "node:fs";
import NetworkController from "../core/NetworkController";

const modelJSON = readFileSync("src/weights/xor.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

const in1 = new Float32Array([1, 0]);
const in2 = new Float32Array([0, 0]);
const in3 = new Float32Array([1, 1]);
const in4 = new Float32Array([0, 1]);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forward(in1));
console.log("Result of [0, 0]: " + network.forward(in2));
console.log("Result of [1, 1]: " + network.forward(in3));
console.log("Result of [0, 1]: " + network.forward(in4));
console.log("-----------------------------------------");
