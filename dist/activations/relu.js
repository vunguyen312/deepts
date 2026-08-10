"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const neuralNetwork_1 = require("../neuralNetwork");
class ReLUNeuron extends neuralNetwork_1.BaseNeuron {
    activation(x) {
        return Math.max(0, x);
    }
    activationDerivative(x) {
        if (x > 0) {
            return 1;
        }
        return 0;
    }
}
class ReLULayer extends neuralNetwork_1.BaseLayer {
    getActivationFunction() {
        return "relu";
    }
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new ReLUNeuron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}
exports.default = ReLULayer;
//# sourceMappingURL=relu.js.map