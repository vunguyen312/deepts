import { BaseNeuron, BaseLayer } from "../neuralNetwork";

class ReLUNeuron extends BaseNeuron {
    protected activation(x: number): number {
        return Math.max(0, x);
    }

    protected activationDerivative(x: number): number {
        if (x > 0) {
            return 1;
        }
        return 0;
    }
}

export default class ReLULayer extends BaseLayer {
    protected getActivationFunction(): string {
        return "relu";
    }

    protected spawnNeurons(): ReLUNeuron[] {
        const neurons: ReLUNeuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new ReLUNeuron(this.inputSize, this.outputSize);
            neurons.push(neuron);
        }
        return neurons;
    }
}