import { BaseNeuron, BaseLayer, BaseNetwork } from "./baseNetwork";

class Perceptron extends BaseNeuron {
    private step(x: number): number {
        if (x <= 0) {
            return 0;
        }
        return 1;
    }

    protected activation(x: number): number {
        return this.step(x);
    }

    protected activationDerivative(x: number): number {
        return 0;
    }
}

class PerceptronLayer extends BaseLayer<Perceptron> {
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

class PerceptronNetwork extends BaseNetwork<Perceptron> {}

class SigmoidNeuron extends BaseNeuron {
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    private sigmoidDerivative(x: number): number {
        const sigmoidValue = this.sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }

    protected activation(x: number): number {
        return this.sigmoid(x);
    }

    protected activationDerivative(x: number): number {
        return this.sigmoidDerivative(x);
    }
}

class SigmoidLayer extends BaseLayer<SigmoidNeuron> {
    protected getActivationFunction(): string {
        return "sigmoid";
    }

    protected spawnNeurons(): SigmoidNeuron[] {
        const neurons: SigmoidNeuron[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const neuron = new SigmoidNeuron(this.inputSize, this.outputSize);
            console.log("WEIGHTS: " + neuron.getWeights());
            console.log("BIAS: " + neuron.getBias());
            neurons.push(neuron);
        }
        console.log("NEURONS IN LAYER: " + neurons.length);
        return neurons;
    }
}

class SigmoidNetwork extends BaseNetwork<SigmoidNeuron> {}

const layer1 = new SigmoidLayer(2, 3);
const layer2 = new SigmoidLayer(3, 1);

const network = new SigmoidNetwork([layer1, layer2], 0.1);

for (let i = 0; i < 20000; i++) {
    network.train([0, 1], [1]);
    network.train([1, 1], [0]);
    network.train([1, 0], [1]);
    network.train([0, 0], [0]);
}

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
network.freezeToJSON();