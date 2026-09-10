import { Tensor } from "../math/Tensor";

type LayerType = "linear" | "activation"

export interface FrozenLayer {
    layerType: LayerType;
}

export interface FrozenLinear extends FrozenLayer {
    inputSize: number;
    outputSize: number;
    weights: number[];
    biases: number[];
}

export interface Parameters {
    gradWeights: Tensor;
    gradBiases: Tensor;
    weights: Tensor;
    biases: Tensor;
}

export abstract class Layer {
    public abstract forward(inputs: Tensor): Tensor;
    public abstract backward(errors: Tensor): Tensor;
    public abstract freeze(): FrozenLayer;
    public get params(): Parameters | null {
        return null;
    } 
}

export class LinearLayer extends Layer {
    private inputSize: number;
    private outputSize: number;
    private _params: Parameters;
    private inputs: Tensor;

    constructor(inputSize: number, outputSize: number, weights?: Float32Array,
                biases?: Float32Array) {
        if (inputSize <= 0) {
            throw new Error("Layer must have one or more inputs.");
        }

        if (outputSize <= 0) {
            throw new Error("Layer must have one or more outputs");
        }

        super();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this._params = this.generateParameters(weights!, biases!);
        this.inputs = Tensor.zeros(this.inputSize);
    }

    private generateParameters(weights: Float32Array, 
                               biases: Float32Array): Parameters {
        const layerShape = [this.outputSize, this.inputSize];
        if (weights && biases) {
            return {
                gradWeights: Tensor.zeros(...layerShape),
                gradBiases: Tensor.zeros(this.outputSize),
                weights: new Tensor(weights, layerShape),
                biases: new Tensor(biases, [this.outputSize]),
            };
        }
        return {
            gradWeights: Tensor.zeros(...layerShape),
            gradBiases: Tensor.zeros(this.outputSize),
            weights: Tensor.xavier(this.inputSize, this.outputSize, 
                                   ...layerShape),
            biases: Tensor.rand(this.outputSize),
        };
    }

    private compute(): Tensor {
        const { weights } = this._params;
        const weightsT = weights.transpose();
        const computedResult = this.inputs.matmul(weightsT);
        computedResult.adds(this._params.biases);
        return computedResult;
    }

    private accumulateGradBiases(deltas: Tensor): void {
        const { gradBiases } = this._params;
        const colSize = deltas.shape[0];
        for (let i = 0; i < deltas.data.length; i++) {
            // may need change later bcuz of transpose
            gradBiases.data[i % colSize] += deltas.data[i];
        } 
    }

    private accumulateGrad(deltas: Tensor): void {
        const { gradWeights } = this._params;

        const VECTOR_DIMS = 1;
        if (this.inputs.shape.length === VECTOR_DIMS) {
            this.inputs.reshapes(VECTOR_DIMS, this.inputSize);
        }
        const deltasCol = new Tensor(deltas.data, deltas.shape);
        if (deltasCol.shape.length === VECTOR_DIMS) {
            deltasCol.reshapes(VECTOR_DIMS, this.outputSize);
        }

        deltasCol.transposes();
        const weightGrad = deltasCol.matmul(this.inputs);
        gradWeights.adds(weightGrad);
        this.accumulateGradBiases(deltasCol);
    }

    public forward(inputs: Tensor): Tensor {
        this.inputs = new Tensor(inputs.data, inputs.shape);
        return this.compute();
    }

    public backward(deltas: Tensor): Tensor {
        const weights = this._params.weights;
        const prevErrors = deltas.matmul(weights);
        this.accumulateGrad(deltas);

        return prevErrors;
    }
    
    public freeze(): FrozenLinear {
        return {
            layerType: "linear",
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            weights: [...this._params.weights.data],
            biases: [...this._params.biases.data]
        };
    }

    get params(): Parameters {
        return this._params;
    }
}

export const loadLinear = (layer: FrozenLinear): Layer => {
    const floatWeights = new Float32Array(layer.weights);
    const floatBiases = new Float32Array(layer.biases);
    return new LinearLayer(layer.inputSize, layer.outputSize, floatWeights,
                           floatBiases);
}

