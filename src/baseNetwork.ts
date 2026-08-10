import { Vector, Matrix } from './math';

interface FrozenNeuron {
    weights: number[];
    bias: number;
}

interface FrozenLayer {
    activationFunction: string;
    inputSize: number;
    outputSize: number;
    neurons: FrozenNeuron[];
}

interface FrozenNetwork {
    learningRate: number;
    layers: FrozenLayer[];
}

export abstract class BaseNeuron {
    protected inputs: number[];
    protected weights: number[];
    protected bias: number;
    protected weightedSum: number;

    constructor(inputSize: number, outputSize: number) {
        this.inputs = [];
        this.weights = Vector.randomVector(inputSize, outputSize);
        this.bias = Math.random();
        this.weightedSum = 0;
    }

    protected abstract activation(x: number): number;

    protected abstract activationDerivative(x: number): number;

    public computeDelta(error: number): number {
        return error * this.activationDerivative(this.weightedSum);
    }

    public updateParams(learningRate: number, delta: number): void {
        const updateStep = Vector.scalarMul(learningRate * delta, this.inputs);
        this.weights = Vector.add(this.weights, updateStep);
        this.bias += learningRate * delta;
    }

    public compute(inputs: number[]): number {
        this.inputs = inputs;
        this.weightedSum = Vector.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation(this.weightedSum);
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

export abstract class BaseLayer<T extends BaseNeuron> {
    protected readonly activationFunction: string;
    protected inputSize: number;
    protected outputSize: number;
    protected neurons: T[];
    
    constructor(inputSize: number, outputSize: number) {
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

    protected abstract getActivationFunction(): string;

    protected abstract spawnNeurons(): T[];

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
            activationFunction: this.activationFunction,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            neurons: this.neurons.map(neuron => neuron.freeze())
        };
    }
}

export abstract class BaseNetwork<T extends BaseNeuron> {
    protected layers: BaseLayer<T>[];
    protected learningRate: number;

    constructor(layers: BaseLayer<T>[], learningRate: number) {
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