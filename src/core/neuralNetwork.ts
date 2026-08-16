import { Vector, Matrix } from "../math/math";
import { Activation, ActivationFunc, activationMap } from "../math/activations";

interface FrozenNeuron {
    weights: number[];
    bias: number;
}

export interface FrozenLayer {
    activation: ActivationFunc;
    inputSize: number;
    outputSize: number;
    neurons: FrozenNeuron[];
}

export interface FrozenNetwork {
    layers: FrozenLayer[];
}

export class Neuron {
    private readonly activation: Activation;
    private inputs: Float32Array;
    private gradWeights: Float32Array;
    private gradBias: number;
    private weights: Float32Array;
    private bias: number;
    private weightedSum: number;

    constructor(activation: Activation, weights: Float32Array, bias: number);

    constructor(activation: Activation, inputSize: number, outputSize: number);

    constructor(activation: Activation, 
                arg1: number | Float32Array, arg2: number) {
        this.activation = activation;
        this.inputs = new Float32Array();
        this.weightedSum = 0;
        this.gradBias = 0;

        if (typeof arg1 === 'number') {
            this.gradWeights = new Float32Array(arg1);
            this.weights = Vector.randomVector(arg1, arg2);
            this.bias = Math.random();
            return;
        }
        this.gradWeights = new Float32Array(arg1.length);
        this.weights = arg1;
        this.bias = arg2;
    }

    public computeDelta(error: number): number {
        return error * this.activation.derivative(this.weightedSum);
    }

    public acculumateGrad(delta: number): void {
        const gradient = Vector.scalarMul(delta, this.inputs);
        this.gradWeights = Vector.add(this.gradWeights, gradient);
        this.gradBias += delta;
    }

    public step(learningRate: number): void {
        const updateStep = Vector.scalarMul(learningRate, this.gradWeights);
        this.weights = Vector.add(this.weights, updateStep);
        this.bias += learningRate * this.gradBias;
    }

    public zeroGrad(): void {
        for (let i = 0; i < this.gradWeights.length; i++) {
            this.gradWeights[i] = 0;
        }
        this.gradBias = 0;
    }

    public compute(inputs: Float32Array): number {
        this.inputs = inputs;
        this.weightedSum = Vector.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation.fn(this.weightedSum);
        return activationValue;
    }

    public getWeights(): Float32Array {
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
            weights: Array.from(this.weights),
            bias: this.bias
        };
    }
}

export class Layer {
    private readonly activation: ActivationFunc;
    private inputSize: number;
    private outputSize: number;
    private neurons: Neuron[];

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

    private spawnNeurons(): Neuron[] {
        const neurons: Neuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Neuron(activationMap[this.activation], 
                                      this.inputSize, 
                                      this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }

    public forward(inputs: Float32Array): Float32Array {
        const result: number[] = [];
        for (const neuron of this.neurons) {
            const computeNeuron = neuron.compute(inputs);
            result.push(computeNeuron);
        }
        return new Float32Array(result);
    }

    public backward(errors: Float32Array): Float32Array {
        const deltas = this.neurons.map((neuron, i) => 
            neuron.computeDelta(errors[i])
        );
        const deltaFloat = [new Float32Array(deltas)];

        const weights = this.getLayerWeights();
        const prevErrorsMat = Matrix.mul(weights, deltaFloat);
        const prevErrors = prevErrorsMat[0];

        this.neurons.map((neuron, i) => 
            neuron.acculumateGrad(deltas[i])
        );

        return prevErrors;
    }

    private getLayerWeights(): Float32Array[] {
        return this.neurons.map(neuron => 
            neuron.getWeights()
        );
    }

    public getNeurons(): Neuron[] {
        return this.neurons;
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
    private layers: Layer[];

    constructor(layers: Layer[]) {
        if (layers.length === 0) {
            throw new Error("Network must have at least one layer.");
        }

        this.layers = [...layers];
    }

    public forward(inputs: Float32Array): Float32Array {
        let valuePassed: Float32Array = new Float32Array(inputs);

        for (const layer of this.layers) {
            valuePassed = layer.forward(valuePassed);
        }

        return valuePassed;
    }

    public backward(inputData: Float32Array, expectedOutput: Float32Array): void {
        const output = this.forward(inputData);

        let errors: Float32Array = expectedOutput.map(
            (target, i) => target - output[i]
        );
        for (let i = this.layers.length - 1; i >= 0; i--) {
            errors = this.layers[i].backward(errors);
        }
    }

    public getNeurons(): Neuron[] {
        const result: Neuron[] = [];
        for (const layer of this.layers) {
            const layerNeurons = layer.getNeurons();
            for (const neuron of layerNeurons) {
                result.push(neuron);
            }
        }

        return result;
    }

    public freeze(): FrozenNetwork {
        return {
            layers: this.layers.map(layer => layer.freeze())
        };
    }
}