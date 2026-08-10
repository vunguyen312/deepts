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
export declare abstract class BaseNeuron {
    protected inputs: number[];
    protected weights: number[];
    protected bias: number;
    protected weightedSum: number;
    constructor(inputSize: number, outputSize: number);
    protected abstract activation(x: number): number;
    protected abstract activationDerivative(x: number): number;
    computeDelta(error: number): number;
    updateParams(learningRate: number, delta: number): void;
    compute(inputs: number[]): number;
    getWeights(): number[];
    getBias(): number;
    getWeightedSum(): number;
    freeze(): FrozenNeuron;
}
export declare abstract class BaseLayer {
    protected readonly activationFunction: string;
    protected inputSize: number;
    protected outputSize: number;
    protected neurons: BaseNeuron[];
    constructor(inputSize: number, outputSize: number);
    protected abstract getActivationFunction(): string;
    protected abstract spawnNeurons(): BaseNeuron[];
    forward(inputs: number[]): number[];
    backward(learningRate: number, errors: number[]): number[];
    private getLayerWeights;
    freeze(): FrozenLayer;
}
export declare class NeuralNetwork {
    protected layers: BaseLayer[];
    protected learningRate: number;
    constructor(layers: BaseLayer[], learningRate: number);
    forwardPass(inputs: number[]): number[];
    train(inputData: number[], expectedOutput: number[]): void;
    freeze(): FrozenNetwork;
}
export {};
//# sourceMappingURL=neuralNetwork.d.ts.map