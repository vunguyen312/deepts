abstract class BaseNeuron {
    protected weights: number[];
    protected bias: number;

    constructor(weights: number[], bias: number) {
        this.weights = [...weights];
        this.bias = bias;
    }

    protected abstract activation(x: number): number;

    protected dot(vec1: number[], vec2: number[]) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    public compute(inputs: number[]): number {
        const linearCombination = this.dot(inputs, this.weights) + this.bias;
        const activationValue = this.activation(linearCombination);
        return activationValue;
    }

    public getWeights(): number[] {
        return [...this.weights];
    }

    public getBias(): number {
        return this.bias;
    }
}

abstract class BaseNetwork<T extends BaseNeuron> {
    protected layers: T[][];

    constructor(layers: T[][]) {
        if (layers.length === 0) {
            throw new Error("Network must have at least one layer.");
        }

        this.layers = layers.map(layer => {
            if (layer.length === 0) {
                throw new Error('Layer cannot be empty.');
            }
            return [...layer];
        });
    }

    public forwardPass(inputs: number[]): number[] {
        let valuePassed = [...inputs];

        for (let i = 0; i < this.layers.length; i++) {
            const newInputs: number[] = [];
            for (let j = 0; j < this.layers[i].length; j++) {
                const currNeuron = this.layers[i][j];
                const actionValue = currNeuron.compute(valuePassed);
                newInputs.push(actionValue);
            }
            valuePassed = newInputs;
        }

        return valuePassed;
    }
}

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
}

class PerceptronNetwork extends BaseNetwork<Perceptron> {
    
}

class SigmoidNeuron extends BaseNeuron {
    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    protected activation(x: number): number {
        return this.sigmoid(x);
    }
}

class SigmoidNetwork extends BaseNetwork<SigmoidNeuron> {

}



const layers = [[new Perceptron([-1, -1], 7), new Perceptron([2, 2], -6), new Perceptron([2, -2], 6), new Perceptron([5, -2], -20)], 
                [new Perceptron([2, 2, 2, 2], -7)]];

const perceptronNetwork = new PerceptronNetwork(layers);
console.log(perceptronNetwork.forwardPass([5, 1]));

