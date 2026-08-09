"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const baseNetwork_1 = require("./baseNetwork");
class Perceptron extends baseNetwork_1.BaseNeuron {
    step(x) {
        if (x <= 0) {
            return 0;
        }
        return 1;
    }
    activation(x) {
        return this.step(x);
    }
    activationDerivative(x) {
        return 0;
    }
}
class PerceptronLayer extends baseNetwork_1.BaseLayer {
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
class PerceptronNetwork extends baseNetwork_1.BaseNetwork {
}
class SigmoidNeuron extends baseNetwork_1.BaseNeuron {
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    sigmoidDerivative(x) {
        const sigmoidValue = this.sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
    activation(x) {
        return this.sigmoid(x);
    }
    activationDerivative(x) {
        return this.sigmoidDerivative(x);
    }
}
class SigmoidLayer extends baseNetwork_1.BaseLayer {
    getActivationFunction() {
        return "sigmoid";
    }
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new SigmoidNeuron(this.inputSize, this.outputSize);
            console.log("WEIGHTS: " + neuron.getWeights());
            console.log("BIAS: " + neuron.getBias());
            neurons.push(neuron);
        }
        console.log("NEURONS IN LAYER: " + neurons.length);
        return neurons;
    }
}
class SigmoidNetwork extends baseNetwork_1.BaseNetwork {
}
const layer1 = new SigmoidLayer(2, 3);
const layer2 = new SigmoidLayer(3, 1);
const network = new SigmoidNetwork([layer1, layer2], 0.1);
for (let i = 0; i < 20000; i++) {
    network.train([0, 1], [1]);
    network.train([1, 1], [0]);
    network.train([1, 0], [1]);
    network.train([0, 0], [0]);
}
console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
network.freezeToJSON();
//# sourceMappingURL=index.js.map