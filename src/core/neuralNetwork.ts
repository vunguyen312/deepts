import { Tensor } from "../math/Tensor";
import { Activation, ActivationFunc, activationMap } from "../math/activations";

interface FrozenLayer {
    activation: ActivationFunc;
    inputSize: number;
    outputSize: number;
    weights: number[];
    biases: number[];
}

export interface FrozenNetwork {
    layers: FrozenLayer[];
}

export interface Parameters {
    gradWeights: Tensor;
    gradBiases: Tensor;
    weights: Tensor;
    biases: Tensor;
}

export class Layer {
    private readonly activation: Activation;
    private inputSize: number;
    private outputSize: number;
    private params: Parameters;
    private inputs: Tensor;
    private weightedSums: Tensor;

    constructor(activation: ActivationFunc, inputSize: number, 
                outputSize: number, weights?: Float32Array, 
                biases?: Float32Array) {
        if (inputSize <= 0) {
            throw new Error("Layer must have one or more inputs.");
        }

        if (outputSize <= 0) {
            throw new Error("Layer must have one or more outputs");
        }

        this.activation = activationMap[activation];
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.params = this.generateParameters(weights!, biases!);
        this.inputs = Tensor.zeros([this.outputSize, this.inputSize]);
        this.weightedSums = Tensor.zeros([this.outputSize])
    }

    private generateParameters(weights: Float32Array, 
                               biases: Float32Array): Parameters {
        if (weights && biases) {
            return {
                gradWeights: Tensor.zeros([this.outputSize, this.inputSize]),
                gradBiases: Tensor.zeros([this.outputSize]),
                weights: new Tensor(weights, [this.outputSize, this.inputSize]),
                biases: new Tensor(biases, [this.outputSize]),
            };
        }
        return {
            gradWeights: Tensor.zeros([this.outputSize, this.inputSize]),
            gradBiases: Tensor.zeros([this.outputSize]),
            weights: Tensor.xavier([this.outputSize, this.inputSize], 
                                   this.inputSize, this.outputSize),
            biases: Tensor.rand([this.outputSize]),
        };
    }

    private computeDeltas(errors: Tensor): Tensor {
        const activationDerivs = this.weightedSums.map(element => 
            this.activation.derivative(element)
        );
        return errors.mul(activationDerivs);
    }

    private compute(inputs: Tensor): Tensor {
        this.inputs = new Tensor(inputs.data, inputs.shape);
        const { weights } = this.params;
        const computedResult = weights.matmul(inputs);
        computedResult.adds(this.params.biases);
        this.weightedSums = new Tensor(computedResult.data, 
                                       computedResult.shape);
        computedResult.maps(element => this.activation.fn(element));
        return computedResult;
    }

    private accumulateGrad(deltas: Tensor): void {
        const { gradWeights, gradBiases } = this.params;
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                const index = i * this.inputSize + j;
                gradWeights.data[index] += deltas.data[i] * this.inputs.data[j];
            }
            gradBiases.data[i] += deltas.data[i];
        }
    }

    public forward(inputs: Tensor): Tensor {
        return this.compute(inputs);
    }

    public backward(errors: Tensor): Tensor {
        const deltas = this.computeDeltas(errors);

        const weights = this.params.weights;
        const prevErrors = deltas.matmul(weights);

        this.accumulateGrad(deltas);

        return prevErrors;
    }
    
    public freeze(): FrozenLayer {
        return {
            activation: this.activation.id,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            weights: [...this.params.weights.data],
            biases: [...this.params.biases.data]
        };
    }

    get getParams(): Parameters {
        return this.params;
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

    public forward(inputs: Tensor): Tensor {
        let valuePassed = new Tensor(inputs.data, inputs.shape);

        for (const layer of this.layers) {
            valuePassed = layer.forward(valuePassed);
        }

        return valuePassed;
    }

    public backward(inputData: Tensor, expectedOutput: Tensor): void {
        const output = this.forward(inputData);

        let errors = expectedOutput.sub(output);
        for (let i = this.layers.length - 1; i >= 0; i--) {
            errors = this.layers[i].backward(errors);
        }
    }
    
    public freeze(): FrozenNetwork {
        return {
            layers: this.layers.map(layer => layer.freeze())
        };
    }

    get params(): Parameters[] {
        const result: Parameters[] = [];
        for (const layer of this.layers) {
            const params = layer.getParams;
            result.push(params);
        }
        return result;
    }
}