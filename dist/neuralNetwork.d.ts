import { Activation, ActivationFunc } from "./math/activations";
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
    learningRate: number;
    layers: FrozenLayer[];
}
export declare class Neuron {
    private readonly activation;
    protected inputs: number[];
    protected weights: number[];
    protected bias: number;
    protected weightedSum: number;
    constructor(activation: Activation, weights: number[], bias: number);
    constructor(activation: Activation, inputSize: number, outputSize: number);
    computeDelta(error: number): number;
    updateParams(learningRate: number, delta: number): void;
    compute(inputs: number[]): number;
    getWeights(): number[];
    getBias(): number;
    getWeightedSum(): number;
    freeze(): FrozenNeuron;
}
export declare class Layer {
    private readonly activation;
    protected inputSize: number;
    protected outputSize: number;
    protected neurons: Neuron[];
    constructor(activation: ActivationFunc, inputSize: number, outputSize: number, neurons?: Neuron[]);
    protected spawnNeurons(): Neuron[];
    forward(inputs: number[]): number[];
    backward(learningRate: number, errors: number[]): number[];
    private getLayerWeights;
    freeze(): FrozenLayer;
}
export declare class NeuralNetwork {
    protected layers: Layer[];
    protected learningRate: number;
    constructor(layers: Layer[], learningRate: number);
    forwardPass(inputs: number[]): number[];
    train(inputData: number[], expectedOutput: number[]): void;
    freeze(): FrozenNetwork;
}
export {};
//# sourceMappingURL=neuralNetwork.d.ts.map