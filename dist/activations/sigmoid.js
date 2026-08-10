"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const neuralNetwork_1 = require("../neuralNetwork");
class SigmoidNeuron extends neuralNetwork_1.BaseNeuron {
    activation(x) {
        return 1 / (1 + Math.exp(-x));
    }
    activationDerivative(x) {
        const sigmoidValue = this.activation(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
}
class SigmoidLayer extends neuralNetwork_1.BaseLayer {
    getActivationFunction() {
        return "sigmoid";
    }
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new SigmoidNeuron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}
exports.default = SigmoidLayer;
//# sourceMappingURL=sigmoid.js.map