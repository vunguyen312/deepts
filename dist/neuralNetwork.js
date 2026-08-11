"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NeuralNetwork = exports.Layer = exports.Neuron = void 0;
const math_1 = require("./math");
const activations_1 = require("./activations");
class Neuron {
    activation;
    inputs;
    weights;
    bias;
    weightedSum;
    constructor(activation, arg1, arg2) {
        this.activation = activation;
        this.inputs = [];
        this.weightedSum = 0;
        if (typeof arg1 === 'number') {
            this.weights = math_1.Vector.randomVector(arg1, arg2);
            this.bias = Math.random();
            return;
        }
        this.weights = arg1;
        this.bias = arg2;
    }
    computeDelta(error) {
        return error * this.activation.derivative(this.weightedSum);
    }
    updateParams(learningRate, delta) {
        const updateStep = math_1.Vector.scalarMul(learningRate * delta, this.inputs);
        this.weights = math_1.Vector.add(this.weights, updateStep);
        this.bias += learningRate * delta;
    }
    compute(inputs) {
        this.inputs = inputs;
        this.weightedSum = math_1.Vector.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation.fn(this.weightedSum);
        return activationValue;
    }
    getWeights() {
        return this.weights;
    }
    getBias() {
        return this.bias;
    }
    getWeightedSum() {
        return this.weightedSum;
    }
    freeze() {
        return {
            weights: this.weights,
            bias: this.bias
        };
    }
}
exports.Neuron = Neuron;
class Layer {
    activation;
    inputSize;
    outputSize;
    neurons;
    constructor(activation, inputSize, outputSize, neurons) {
        if (inputSize <= 0) {
            throw new Error("Layer must have one or more inputs.");
        }
        if (outputSize <= 0) {
            throw new Error("Layer must have one or more outputs");
        }
        this.activation = activation;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.neurons = neurons || this.spawnNeurons();
    }
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Neuron(activations_1.activationMap[this.activation], this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
    forward(inputs) {
        return this.neurons.map(neuron => neuron.compute(inputs));
    }
    backward(learningRate, errors) {
        const deltas = this.neurons.map((neuron, i) => neuron.computeDelta(errors[i]));
        const weights = this.getLayerWeights();
        const prevErrorsMat = math_1.Matrix.mul(weights, [deltas]);
        const prevErrors = prevErrorsMat[0];
        this.neurons.map((neuron, i) => neuron.updateParams(learningRate, deltas[i]));
        return prevErrors;
    }
    getLayerWeights() {
        return this.neurons.map(neuron => neuron.getWeights());
    }
    freeze() {
        return {
            activation: this.activation,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            neurons: this.neurons.map(neuron => neuron.freeze())
        };
    }
}
exports.Layer = Layer;
class NeuralNetwork {
    layers;
    learningRate;
    constructor(layers, learningRate) {
        if (layers.length === 0) {
            throw new Error("Network must have at least one layer.");
        }
        this.layers = [...layers];
        this.learningRate = learningRate;
    }
    forwardPass(inputs) {
        let valuePassed = [...inputs];
        for (const layer of this.layers) {
            valuePassed = layer.forward(valuePassed);
        }
        return valuePassed;
    }
    train(inputData, expectedOutput) {
        const output = this.forwardPass(inputData);
        let errors = expectedOutput.map((target, i) => target - output[i]);
        for (let i = this.layers.length - 1; i >= 0; i--) {
            errors = this.layers[i].backward(this.learningRate, errors);
        }
    }
    freeze() {
        return {
            learningRate: this.learningRate,
            layers: this.layers.map(layer => layer.freeze())
        };
    }
}
exports.NeuralNetwork = NeuralNetwork;
//# sourceMappingURL=neuralNetwork.js.map