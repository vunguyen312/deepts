"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const neuralNetwork_1 = require("./neuralNetwork");
const sigmoid_1 = __importDefault(require("./activations/sigmoid"));
const relu_1 = __importDefault(require("./activations/relu"));
const layer1 = new sigmoid_1.default(2, 3);
const layer2 = new relu_1.default(3, 1);
const network = new neuralNetwork_1.NeuralNetwork([layer1, layer2], 0.1);
for (let i = 0; i < 20000; i++) {
    network.train([0, 1], [1]);
    network.train([1, 1], [0]);
    network.train([1, 0], [1]);
    network.train([0, 0], [0]);
}
console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
//# sourceMappingURL=index.js.map