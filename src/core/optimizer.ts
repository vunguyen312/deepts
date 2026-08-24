import { Parameters } from "./neuralNetwork";

export abstract class Optimizer {
    protected readonly networkParams: Parameters[];
    protected learningRate: number;

    constructor(networkParams: Parameters[], learningRate: number) {
        this.networkParams = networkParams;
        this.learningRate = learningRate;
    }

    private zeroGradParams({ gradWeights, gradBiases }: Parameters): void {
        gradWeights.zero();
        gradBiases.zero();
    }

    public zeroGrad(): void {
        this.networkParams.forEach(params => this.zeroGradParams(params));
    }
}

export class SGD extends Optimizer {
    private stepParams(params: Parameters): void {
        for (let i = 0; i < params.weights.data.length; i++) {
            const updateStep = this.learningRate * params.gradWeights.data[i];
            params.weights.data[i] += updateStep;
        }
        for (let i = 0; i < params.biases.data.length; i++) {
            const updateStep = this.learningRate * params.gradBiases.data[i];
            params.biases.data[i] += updateStep;
        }
    }

    public step(): void {
        this.networkParams.forEach(params => this.stepParams(params));
    }
}