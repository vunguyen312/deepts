import { BaseNeuron, BaseLayer } from "../neuralNetwork";

class Perceptron extends BaseNeuron {
    protected activation(x: number): number {
        if (x <= 0) {
            return 0;
        }
        return 1;
    }

    protected activationDerivative(x: number): number {
        return 0;
    }
}

export default class PerceptronLayer extends BaseLayer {
    protected getActivationFunction(): string {
        return "step";
    }

    protected spawnNeurons(): Perceptron[] {
        const neurons: Perceptron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new Perceptron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }

        return neurons;
    }
}