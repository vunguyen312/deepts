import { NeuralNetwork } from "./neuralNetwork";
import SigmoidLayer from "./activations/sigmoid";
import ReLULayer from "./activations/relu";

const layer1 = new SigmoidLayer(2, 3);
const layer2 = new ReLULayer(3, 1);

const network = new NeuralNetwork([layer1, layer2], 0.1);

for (let i = 0; i < 20000; i++) {
    network.train([0, 1], [1]);
    network.train([1, 1], [0]);
    network.train([1, 0], [1]);
    network.train([0, 0], [0]);
}

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));