import { Tensor } from "../math/Tensor";
import { FrozenLayer, Layer, Parameters } from "./layers";

export interface FrozenNetwork {
    layers: FrozenLayer[];
}

export class NeuralNetwork {
    private layers: Layer[];

    constructor(layers: Layer[]) {
        if (layers.length === 0) {
            throw new Error("Network must have at least one layer.");
        }

        this.layers = layers;
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

    public get params(): Parameters[] {
        const result: Parameters[] = [];
        for (const layer of this.layers) {
            const params = layer.params;
            if (params) {
                result.push(params);
            }
        }
        return result;
    }
}