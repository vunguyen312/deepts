"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NeuralNetwork = exports.BaseLayer = exports.BaseNeuron = void 0;
const math_1 = require("./math");
class BaseNeuron {
    inputs;
    weights;
    bias;
    weightedSum;
    constructor(inputSize, outputSize) {
        this.inputs = [];
        this.weights = math_1.Vector.randomVector(inputSize, outputSize);
        this.bias = Math.random();
        this.weightedSum = 0;
    }
    computeDelta(error) {
        return error * this.activationDerivative(this.weightedSum);
    }
    updateParams(learningRate, delta) {
        const updateStep = math_1.Vector.scalarMul(learningRate * delta, this.inputs);
        this.weights = math_1.Vector.add(this.weights, updateStep);
        this.bias += learningRate * delta;
    }
    compute(inputs) {
        this.inputs = inputs;
        this.weightedSum = math_1.Vector.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation(this.weightedSum);
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
exports.BaseNeuron = BaseNeuron;
class BaseLayer {
    activationFunction;
    inputSize;
    outputSize;
    neurons;
    constructor(inputSize, outputSize) {
        if (inputSize <= 0) {
            throw new Error("Layer must have one or more inputs.");
        }
        if (outputSize <= 0) {
            throw new Error("Layer must have one or more outputs");
        }
        this.activationFunction = this.getActivationFunction();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.neurons = this.spawnNeurons();
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
            activationFunction: this.activationFunction,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            neurons: this.neurons.map(neuron => neuron.freeze())
        };
    }
}
exports.BaseLayer = BaseLayer;
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