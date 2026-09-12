import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Tensor, loadNetwork } from "../src/index";

const modelJSON = readFileSync(join(__dirname, "./weights/xor.json"), "utf-8");
const modelData = JSON.parse(modelJSON);
const network = loadNetwork(modelData);

const in1 = new Tensor([1, 0]);
const in2 = new Tensor([0, 0]);
const in3 = new Tensor([1, 1]);
const in4 = new Tensor([0, 1]);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forward(in1).data);
console.log("Result of [0, 0]: " + network.forward(in2).data);
console.log("Result of [1, 1]: " + network.forward(in3).data);
console.log("Result of [0, 1]: " + network.forward(in4).data);
console.log("-----------------------------------------");
