import { readFileSync } from "node:fs";
import { join } from "node:path";
import NetworkController from "../core/NetworkController";

// Loads a frozen network and runs inference on all four XOR inputs.
// Paths are resolved relative to this file, so the example works from any CWD.

const modelPath = join(__dirname, "../weights/xor.json");
const modelJSON = readFileSync(modelPath, "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

const in1 = new Float32Array([1, 0]);
const in2 = new Float32Array([0, 0]);
const in3 = new Float32Array([1, 1]);
const in4 = new Float32Array([0, 1]);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forwardPass(in1));
console.log("Result of [0, 0]: " + network.forwardPass(in2));
console.log("Result of [1, 1]: " + network.forwardPass(in3));
console.log("Result of [0, 1]: " + network.forwardPass(in4));
console.log("-----------------------------------------");
