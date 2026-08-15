import { readFileSync } from "node:fs";
import { join } from "node:path";
import NetworkController from "../core/NetworkController";

// Loads a frozen network and runs inference on all four XOR inputs.
// Paths are resolved relative to this file, so the example works from any CWD.

const modelPath = join(__dirname, "../weights/xor.json");
const modelJSON = readFileSync(modelPath, "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forwardPass([1, 0]));
console.log("Result of [0, 0]: " + network.forwardPass([0, 0]));
console.log("Result of [1, 1]: " + network.forwardPass([1, 1]));
console.log("Result of [0, 1]: " + network.forwardPass([0, 1]));
console.log("-----------------------------------------");
