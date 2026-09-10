import { Layer } from "../core/layers";
import type { FrozenLayer } from "../core/layers";
import { Tensor } from "../math/Tensor";
import { relu, sigmoid, step, tanh, softplus } from "./activationFunctions";
import type { Activation, ActivationFunc } from "./activationFunctions";

export interface FrozenActivation extends FrozenLayer {
    activation: ActivationFunc;
}

export abstract class ActivationLayer extends Layer {
    private readonly activation: Activation;
    private preActivation: Tensor;

    constructor(activation: Activation) {
        super();
        this.activation = activation;
        this.preActivation = Tensor.zeros(1);
    }

    public forward(inputs: Tensor): Tensor {
        this.preActivation = new Tensor(inputs.data, inputs.shape);
        inputs.maps(element => this.activation.fn(element));
        return inputs;
    }

    public backward(errors: Tensor): Tensor {
        const activationDerivs = this.preActivation.map(element => 
            this.activation.derivative(element)
        );
        const deltas = errors.mul(activationDerivs);
        return deltas;
    }

    public freeze(): FrozenActivation {
        return {
            layerType: "activation",
            activation: this.activation.id
        };
    }
}

export class Step extends ActivationLayer {
    constructor() {
        super(step);
    }
}

export class ReLU extends ActivationLayer {
    constructor() {
        super(relu);
    }
}

export class Sigmoid extends ActivationLayer {
    constructor() {
        super(sigmoid);
    }
}

export class Tanh extends ActivationLayer {
    constructor() {
        super(tanh);
    }
}

export class Softplus extends ActivationLayer {
    constructor() {
        super(softplus);
    }
}

export const loadActivation = (layer: FrozenActivation): Layer => {
    switch(layer.activation) {
        case "step":
            return new Step();
        case "relu":
            return new ReLU();
        case "sigmoid":
            return new Sigmoid();
        case "tanh":
            return new Tanh();
        case "softplus":
            return new Softplus();
    }
}
