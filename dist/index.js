"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class Vector {
    static randomVector(fanIn, fanOut) {
        const a = Math.sqrt(6 / (fanIn + fanOut));
        const result = [];
        for (let i = 0; i < fanIn; i++) {
            result.push(Math.random() * 2 * a - a);
        }
        return result;
    }
    static dot(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }
        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }
        return sum;
    }
    static scalarMul(scalar, vec) {
        const result = [...vec];
        for (let i = 0; i < vec.length; i++) {
            result[i] *= scalar;
        }
        return result;
    }
    static add(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }
        const result = [...vec1];
        for (let i = 0; i < vec1.length; i++) {
            result[i] += vec2[i];
        }
        return result;
    }
}
class Matrix {
    static zeroMat(rowLength, colLength) {
        const result = [];
        for (let i = 0; i < colLength; i++) {
            const row = new Array(rowLength).fill(0);
            result.push(row);
        }
        return result;
    }
    static mul(mat1, mat2) {
        const rowLength = mat2[0].length;
        const colLength = mat1.length;
        const result = Matrix.zeroMat(rowLength, colLength);
        for (let i = 0; i < rowLength; i++) {
            const ithCol = Matrix.getCol(mat2, i);
            for (let j = 0; j < colLength; j++) {
                result[j][i] = Vector.dot(mat1[j], ithCol);
            }
        }
        return result;
    }
    static transpose(mat) {
        const rowLength = mat[0].length;
        const colLength = mat.length;
        const result = Matrix.zeroMat(colLength, rowLength);
        for (let i = 0; i < rowLength; i++) {
            for (let j = 0; j < colLength; j++) {
                result[i][j] = mat[j][i];
            }
        }
        return result;
    }
    static getCol(mat, j) {
        const result = [];
        for (let i = 0; i < mat.length; i++) {
            result.push(mat[i][j]);
        }
        return result;
    }
}
class BaseNeuron {
    inputs;
    weights;
    bias;
    weightedSum;
    constructor(inputSize, outputSize) {
        this.inputs = [];
        this.weights = Vector.randomVector(inputSize, outputSize);
        this.bias = Math.random();
        this.weightedSum = 0;
    }
    computeDelta(error) {
        return error * this.activationDerivative(this.weightedSum);
    }
    updateParams(learningRate, delta) {
        const updateStep = Vector.scalarMul(learningRate * delta, this.inputs);
        this.weights = Vector.add(this.weights, updateStep);
        this.bias += learningRate * delta;
    }
    compute(inputs) {
        this.inputs = inputs;
        this.weightedSum = Vector.dot(inputs, this.weights) + this.bias;
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
}
class BaseLayer {
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
        const weightsT = Matrix.transpose(weights);
        const deltaCol = deltas.map(d => [d]);
        const prevErrorsMat = Matrix.mul(weightsT, deltaCol);
        const prevErrors = prevErrorsMat.map(row => row[0]);
        this.neurons.map((neuron, i) => neuron.updateParams(learningRate, deltas[i]));
        return prevErrors;
    }
    getLayerWeights() {
        return this.neurons.map(neuron => neuron.getWeights());
    }
}
class BaseNetwork {
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
}
class Perceptron extends BaseNeuron {
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
class PerceptronLayer extends BaseLayer {
    spawnNeurons() {
        const neurons = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Perceptron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}
class PerceptronNetwork extends BaseNetwork {
}
class SigmoidNeuron extends BaseNeuron {
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
class SigmoidLayer extends BaseLayer {
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
class SigmoidNetwork extends BaseNetwork {
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
//# sourceMappingURL=index.js.map