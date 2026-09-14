import { Layer } from "../core/layers";
import type { FrozenLayer } from "../core/layers";
import { Tensor } from "../math/Tensor";

export interface FrozenActivation extends FrozenLayer {
    activation: ActivationFunc;
}

export abstract class ActivationLayer extends Layer {
    private readonly id: ActivationFunc;
    private preActivation: Tensor;

    constructor(id: ActivationFunc) {
        super();
        this.id = id;
        this.preActivation = Tensor.zeros(1);
    }

    protected abstract fn(x: number): number;

    protected abstract derivative(x: number): number;

    public forward(inputs: Tensor): Tensor {
        this.preActivation = new Tensor(inputs.data, inputs.shape);
        inputs.maps(element => this.fn(element));
        return inputs;
    }

    public backward(errors: Tensor): Tensor {
        const activationDerivs = this.preActivation.map(element => 
            this.derivative(element)
        );
        const deltas = errors.mul(activationDerivs);
        return deltas;
    }

    public freeze(): FrozenActivation {
        return {
            layerType: "activation",
            activation: this.id
        };
    }
}

export class Step extends ActivationLayer {
    constructor() {
        super("step");
    }

    protected fn(x: number): number {
        if (x > 0) {
            return 1;
        }
        return 0;
    }

    protected derivative(x: number): number {
        return 0;
    }
}

export class ReLU extends ActivationLayer {
    constructor() {
        super("relu");
    }

    protected fn(x: number): number {
        return Math.max(0, x);
    }

    protected derivative(x: number): number {
        if (x > 0) {
            return 1;
        }
        return 0;
    }
}

export class Sigmoid extends ActivationLayer {
    constructor() {
        super("sigmoid");
    }

    protected fn(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    protected derivative(x: number): number {
        const sigmoidValue = this.fn(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
}

export class Tanh extends ActivationLayer {
    constructor() {
        super("tanh");
    }

    protected fn(x: number): number {
        return 2 / (1 + Math.exp(-2 * x)) - 1;
    }

    protected derivative(x: number): number {
        return 1 - this.fn(x) ** 2;
    }
}

export class Softplus extends ActivationLayer {
    constructor() {
        super("softplus");
    }
    
    protected fn(x: number): number {
        return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
    }

    protected derivative(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }
}

export class Softmax extends ActivationLayer {
    private outputs: Tensor;

    constructor() {
        super("softmax");
        this.outputs = Tensor.zeros(1);
    }

    protected fn(_x: number): number {
        throw new Error("softmax: Not elementwise, use forward instead.");
    }

    protected derivative(_x: number): number {
        throw new Error("softmax: Not elementwise, use backward instead.");
    }

    public forward(inputs: Tensor): Tensor {
        const { shape, data } = inputs;
        // if u think this is ugly then have no fear ill add sum and stuff to
        // tensor later which is approx when i feel like it loool
        const rowLength = shape[shape.length - 1];
        const rowCount = data.length / rowLength;
        for (let row = 0; row < rowCount; row++) {
            const offset = row * rowLength;
            let max = data[offset];
            for (let i = 1; i < rowLength; i++) {
                max = Math.max(max, data[offset + i]);
            }

            let expSum = 0;
            for (let i = 0; i < rowLength; i++) {
                const exp = Math.exp(data[offset + i] - max);
                data[offset + i] = exp;
                expSum += exp;
            }

            for (let i = 0; i < rowLength; i++) {
                data[offset + i] /= expSum;
            }
        }

        this.outputs = new Tensor(data, shape);
        return inputs;
    }

    public backward(errors: Tensor): Tensor {
        return errors;
    }
}

const activationClasses = {
    step: Step,
    relu: ReLU,
    sigmoid: Sigmoid,
    tanh: Tanh,
    softplus: Softplus,
    softmax: Softmax
};

export type ActivationFunc = keyof typeof activationClasses;

export const loadActivation = (layer: FrozenActivation): ActivationLayer => {
    return new (activationClasses[layer.activation])();
}