import { Neuron } from "./neuralNetwork";

export abstract class Optimizer {
    protected readonly neurons: Neuron[];
    protected learningRate: number;

    constructor(neurons: Neuron[], learningRate: number) {
        this.neurons = neurons;
        this.learningRate = learningRate;
    }
}

export class SGDOptimizer extends Optimizer {
    public step(): void {
        this.neurons.map(neuron => neuron.step(this.learningRate));
    }

    public zeroGrad(): void {
        this.neurons.map(neuron => neuron.zeroGrad());
    }
}