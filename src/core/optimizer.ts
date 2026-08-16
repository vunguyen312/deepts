import { Parameters } from "./neuralNetwork";

export abstract class Optimizer {
    protected readonly networkParams: Parameters[];
    protected learningRate: number;

    constructor(networkParams: Parameters[], learningRate: number) {
        this.networkParams = networkParams;
        this.learningRate = learningRate;
    }

    private zeroGradParams(params: Parameters): void {
        for (let i = 0; i < params.gradWeights.length; i++) {
            params.gradWeights[i] = 0;
        }
        params.gradBias = 0;
    }

    public zeroGrad(): void {
        this.networkParams.forEach(params => this.zeroGradParams(params));
    }
}

export class SGDOptimizer extends Optimizer {
    private stepParams(params: Parameters): void {
        for (let i = 0; i < params.weights.length; i++) {
            params.weights[i] += this.learningRate * params.gradWeights[i];
        }
        params.bias += this.learningRate * params.gradBias;
    }

    public step(): void {
        this.networkParams.forEach(params => this.stepParams(params));
    }
}