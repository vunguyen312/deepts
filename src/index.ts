class Vector {
    public static randomVector(dim: number): number[] {
        const resultVector: number[] = [];
        for (let i = 0; i < dim; i++) {
            const randomValue = Math.random();
            resultVector.push(randomValue);
        }

        return resultVector;
    }

    public static dot(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    public static scalarMul(scalar: number, vec: number[]): number[] {
        const result = [...vec];
        for (let i = 0; i < vec.length; i++) {
            result[i] *= scalar;
        }

        return result;
    }

    public static add(vec1: number[], vec2: number[]): number[] {
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

abstract class BaseNeuron {
    protected inputs: number[];
    protected weights: number[];
    protected bias: number;
    protected weightedSum: number;

    constructor(inputSize: number) {
        this.inputs = [];
        this.weights = Vector.randomVector(inputSize);
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
}

abstract class BaseLayer<T extends BaseNeuron> {
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

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.neurons = this.spawnNeurons();
    }

    protected abstract spawnNeurons(): T[];

    public forward(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.compute(inputs));
    }

    public backward(learningRate: number, errors: number[]): number[] {
        const deltas = this.neurons.map((neuron, i) => 
            neuron.computeDelta(errors[i])
        );

        const prevErrors: number[] = new Array(this.inputSize).fill(0);
        for (let i = 0; i < this.neurons.length; i++) {
            const weights = this.neurons[i].getWeights();
            for (let j = 0; j < this.inputSize; j++) {
                prevErrors[j] += weights[j] * deltas[i];
            }
        }

        this.neurons.map((neuron, i) => 
            neuron.updateParams(learningRate, deltas[i])
        );

        return prevErrors;
    }
}

abstract class BaseNetwork<T extends BaseNeuron> {
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
}

class Perceptron extends BaseNeuron {
    private step(x: number): number {
        if (x <= 0) {
            return 0;
        }
        return 1;
    }

    protected activation(x: number): number {
        return this.step(x);
    }

    protected activationDerivative(x: number): number {
        return 0;
    }
}

class PerceptronLayer extends BaseLayer<Perceptron> {
    protected spawnNeurons(): Perceptron[] {
        const neurons: Perceptron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Perceptron(this.inputSize);
            neurons.push(neuron);
        }

        return neurons;
    }
}

class PerceptronNetwork extends BaseNetwork<Perceptron> {}

class SigmoidNeuron extends BaseNeuron {
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    private sigmoidDerivative(x: number): number {
        const sigmoidValue = this.sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }

    protected activation(x: number): number {
        return this.sigmoid(x);
    }

    protected activationDerivative(x: number): number {
        return this.sigmoidDerivative(x);
    }
}

class SigmoidLayer extends BaseLayer<SigmoidNeuron> {
    protected spawnNeurons(): SigmoidNeuron[] {
        const neurons: SigmoidNeuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new SigmoidNeuron(this.inputSize);
            console.log("WEIGHTS: " + neuron.getWeights());
            console.log("BIAS: " + neuron.getBias());
            neurons.push(neuron);
        }
        console.log("NEURONS IN LAYER: " + neurons.length);
        return neurons;
    }
}

class SigmoidNetwork extends BaseNetwork<SigmoidNeuron> {}

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
