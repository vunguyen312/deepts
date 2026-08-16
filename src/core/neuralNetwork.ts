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

export interface Parameters {
    gradWeights: Float32Array;
    gradBias: number;
    weights: Float32Array;
    bias: number;
}

// TODO: replace with tensor because the object overhead blows up past a 
//       certain network size
export class Neuron {
    private inputs: Float32Array;
    private params!: Parameters;
    private weightedSum: number;

    constructor(weights: Float32Array, bias: number);

    constructor(inputSize: number, outputSize: number);

    constructor(arg1: number | Float32Array, arg2: number) {
        this.inputs = new Float32Array();
        this.weightedSum = 0;

        if (typeof arg1 === 'number') {
            this.params = {
                gradWeights: new Float32Array(arg1),
                gradBias: 0,
                weights: Vector.randomVector(arg1, arg2),
                bias: Math.random()
            };
            return;
        }
        this.params = {
            gradWeights: new Float32Array(arg1.length),
            gradBias: 0,
            weights: arg1,
            bias: arg2
        };
    }

    public getInputs(): Float32Array {
        return this.inputs;
    }

    public getParams(): Parameters {
        return this.params;
    }

    public getWeights(): Float32Array {
        return this.params.weights;
    }

    public getWeightedSum(): number {
        return this.weightedSum;
    }

    public setInputs(inputs: Float32Array): void {
        this.inputs = inputs;
    }

    public setWeightedSum(weightedSum: number): void {
        this.weightedSum = weightedSum;
    }

    public freeze(): FrozenNeuron {
        return {
            weights: Array.from(this.params.weights),
            bias: this.params.bias
        };
    }
}

export class Layer {
    private readonly activation: Activation;
    private inputSize: number;
    private outputSize: number;
    private neurons: Neuron[];

    constructor(activation: ActivationFunc, inputSize: number, 
                outputSize: number, neurons?: Neuron[]) {
        if (inputSize <= 0) {
            throw new Error("Layer must have one or more inputs.");
        }

        if (outputSize <= 0) {
            throw new Error("Layer must have one or more outputs");
        }

        this.activation = activationMap[activation];
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.neurons = neurons || this.spawnNeurons();
    }

    private spawnNeurons(): Neuron[] {
        const neurons: Neuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Neuron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        
        return neurons;
    }

    private computeDelta(error: number, i: number): number {
        const weightedSum = this.neurons[i].getWeightedSum();
        return error * this.activation.derivative(weightedSum);
    }

    private compute(inputs: Float32Array, i: number): number {
        const neuron = this.neurons[i];
        const params = neuron.getParams();
        neuron.setInputs(inputs);
        const computedResult = Vector.dot(inputs, params.weights) 
            + params.bias;
        neuron.setWeightedSum(computedResult);
        const activationValue = this.activation.fn(computedResult);
        return activationValue;
    }

    private acculumatedGrad(delta: number, i: number): void {
        const neuron = this.neurons[i];
        const params = neuron.getParams();
        const inputs = neuron.getInputs();
        for (let j = 0; j < params.gradWeights.length; j++) {
            params.gradWeights[j] += delta * inputs[j];
        }
        params.gradBias += delta;
    }

    public forward(inputs: Float32Array): Float32Array {
        const result = new Float32Array(this.neurons.length);
        for (let i = 0; i < this.neurons.length; i++) {
            const computeResult = this.compute(inputs, i);
            result[i] = computeResult;
        }

        return result;
    }

    public backward(errors: Float32Array): Float32Array {
        const deltas = this.neurons.map((neuron, i) => 
            this.computeDelta(errors[i], i)
        );
        const deltaFloat = [new Float32Array(deltas)];

        const weights = this.getLayerWeights();
        const prevErrorsMat = Matrix.mul(weights, deltaFloat);
        const prevErrors = prevErrorsMat[0];

        this.neurons.forEach((neuron, i) => 
            this.acculumatedGrad(deltas[i], i)
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
            activation: this.activation.id,
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

    public getParams(): Parameters[] {
        const result: Parameters[] = [];
        for (const layer of this.layers) {
            const layerNeurons = layer.getNeurons();
            for (const neuron of layerNeurons) {
                const params = neuron.getParams();
                result.push(params);
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