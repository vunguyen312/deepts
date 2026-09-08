import { Parameters } from "./neuralNetwork";
import { Tensor } from "../math/Tensor";

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
    private readonly momentum: number;
    private weightVelocities: Tensor[];
    private biasVelocities: Tensor[];

    constructor(networkParams: Parameters[], learningRate: number, 
                momentum?: number) {
        super(networkParams, learningRate);
        const DEFAULT_MOMENTUM = 0;
        this.momentum = momentum ?? DEFAULT_MOMENTUM;
        this.weightVelocities = new Array(networkParams.length);
        this.biasVelocities = new Array(networkParams.length);
        this.initializeVelocities();
    }

    private initializeVelocities(): void {
        this.networkParams.forEach((params, i) => {
            this.weightVelocities[i] = Tensor.zeros(...params.weights.shape);
            this.biasVelocities[i] = Tensor.zeros(...params.biases.shape);
        });
    }

    private updateVelocities(params: Parameters, index: number): void {
        const weightVelocity = this.weightVelocities[index];
        weightVelocity.maps((element, i) => 
            this.momentum * element + params.gradWeights.data[i]
        );

        const biasVelocity = this.biasVelocities[index];
        biasVelocity.maps((element, i) =>
            this.momentum * element + params.gradBiases.data[i]
        );
    }

    private stepParams(params: Parameters, index: number): void {
        this.updateVelocities(params, index);

        const weightVelocity = this.weightVelocities[index];
        params.weights.maps((element, i) => 
            element + this.learningRate * weightVelocity.data[i]
        );

        const biasVelocity = this.biasVelocities[index];
        params.biases.maps((element, i) => 
            element + this.learningRate * biasVelocity.data[i]
        );
    }

    public step(): void {
        this.networkParams.forEach((params, i) => this.stepParams(params, i));
    }
}