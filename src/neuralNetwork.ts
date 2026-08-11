import { Vector, Matrix } from "./math/math";
import { Activation, ActivationFunc, activationMap } from "./math/activations";

interface FrozenNeuron {
    weights: number[];
    bias: number;
}

export interface FrozenLayer {
    // Keep to layer level for now. I don't think libs like PyTorch allow
    // for control of activation funcs at the neuron level (?)
    activation: ActivationFunc;
    inputSize: number;
    outputSize: number;
    neurons: FrozenNeuron[];
}

export interface FrozenNetwork {
    learningRate: number;
    layers: FrozenLayer[];
}

export class Neuron {
    private readonly activation: Activation;
    protected inputs: number[];
    protected weights: number[];
    protected bias: number;
    protected weightedSum: number;

    constructor(activation: Activation, weights: number[], bias: number);

    constructor(activation: Activation, inputSize: number, outputSize: number);

    constructor(activation: Activation, 
                arg1: number | number[], arg2: number) {
        this.activation = activation;
        this.inputs = [];
        this.weightedSum = 0;

        if (typeof arg1 === 'number') {
            this.weights = Vector.randomVector(arg1, arg2);
            this.bias = Math.random();
            return;
        }
        this.weights = arg1;
        this.bias = arg2;
    }

    public computeDelta(error: number): number {
        return error * this.activation.derivative(this.weightedSum);
    }

    public updateParams(learningRate: number, delta: number): void {
        const updateStep = Vector.scalarMul(learningRate * delta, this.inputs);
        this.weights = Vector.add(this.weights, updateStep);
        this.bias += learningRate * delta;
    }

    public compute(inputs: number[]): number {
        this.inputs = inputs;
        this.weightedSum = Vector.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation.fn(this.weightedSum);
        return activationValue;
    }

    public getWeights(): number[] {
        return this.weights;
    }

    public getBias(): number {
        return this.bias;
    }

    public getWeightedSum(): number {
        return this.weightedSum;
    }

    public freeze(): FrozenNeuron {
        return {
            weights: this.weights,
            bias: this.bias
        };
    }
}

export class Layer {
    private readonly activation: ActivationFunc;
    protected inputSize: number;
    protected outputSize: number;
    protected neurons: Neuron[];

    constructor(activation: ActivationFunc, inputSize: number, outputSize: number, 
                neurons?: Neuron[]) {
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

    protected spawnNeurons(): Neuron[] {
        const neurons: Neuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Neuron(activationMap[this.activation], 
                                      this.inputSize, 
                                      this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }

    public forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.compute(inputs));
    }

    public backward(learningRate: number, errors: number[]): number[] {
        const deltas = this.neurons.map((neuron, i) => 
            neuron.computeDelta(errors[i])
        );

        const weights = this.getLayerWeights();
        const prevErrorsMat = Matrix.mul(weights, [deltas]);
        const prevErrors = prevErrorsMat[0];

        this.neurons.map((neuron, i) => 
            neuron.updateParams(learningRate, deltas[i])
        );

        return prevErrors;
    }

    private getLayerWeights(): number[][] {
        return this.neurons.map(neuron => 
            neuron.getWeights()
        );
    }

    public freeze(): FrozenLayer {
        return {
            activation: this.activation,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            neurons: this.neurons.map(neuron => neuron.freeze())
        };
    }
}

export class NeuralNetwork {
    protected layers: Layer[];
    protected learningRate: number;

    constructor(layers: Layer[], learningRate: number) {
        if (layers.length === 0) {
            throw new Error("Network must have at least one layer.");
        }

        this.layers = [...layers];
        this.learningRate = learningRate;
    }

    public forwardPass(inputs: number[]): number[] {
        let valuePassed = [...inputs];

        for (const layer of this.layers) {
            valuePassed = layer.forward(valuePassed);
        }

        return valuePassed;
    }

    // TODO: add batching
    public train(inputData: number[], expectedOutput: number[]): void {
        const output = this.forwardPass(inputData);

        let errors = expectedOutput.map((target, i) => target - output[i]);
        for (let i = this.layers.length - 1; i >= 0; i--) {
            errors = this.layers[i].backward(this.learningRate, errors);
        }
    }

    public freeze(): FrozenNetwork {
        return {
            learningRate: this.learningRate,
            layers: this.layers.map(layer => layer.freeze())
        };
    }
}