import { BaseNeuron, BaseLayer } from "../neuralNetwork";

class SigmoidNeuron extends BaseNeuron {
    protected activation(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    protected activationDerivative(x: number): number {
        const sigmoidValue = this.activation(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
}

export default class SigmoidLayer extends BaseLayer {
    protected getActivationFunction(): string {
        return "sigmoid";
    }

    protected spawnNeurons(): SigmoidNeuron[] {
        const neurons: SigmoidNeuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new SigmoidNeuron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}