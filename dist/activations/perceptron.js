"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const neuralNetwork_1 = require("../neuralNetwork");
class Perceptron extends neuralNetwork_1.BaseNeuron {
    activation(x) {
        if (x <= 0) {
            return 0;
        }
        return 1;
    }
    activationDerivative(x) {
        return 0;
    }
}
class PerceptronLayer extends neuralNetwork_1.BaseLayer {
    getActivationFunction() {
        return "step";
    }
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Perceptron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}
exports.default = PerceptronLayer;
//# sourceMappingURL=perceptron.js.map