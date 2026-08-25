export interface Activation {
    readonly id: ActivationFunc;
    readonly fn: (x: number) => number;
    readonly derivative: (x: number) => number;
}

export const step: Activation = {
    id: "step",
    fn: x => (x > 0 ? 1 : 0),
    derivative: _x => 0
};

export const relu: Activation = {
    id: "relu",
    fn: x => Math.max(0, x),
    derivative: x => (x > 0 ? 1 : 0),
};

export const sigmoid: Activation = {
    id: "sigmoid",
    fn: x => 1 / (1 + Math.exp(-x)),
    derivative: x => {
        const sigmoidValue = sigmoid.fn(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
};

export const tanh: Activation = {
    id: "tanh",
    fn: x => 2 / (1 + Math.exp(-2 * x)) - 1,
    derivative: x => 1 - tanh.fn(x) ** 2
}

export const activationMap = {
    step,
    relu,
    sigmoid,
    tanh
} satisfies Record<string, Activation>;
// Builds a mapping for Activations by their strings
export type ActivationFunc = keyof typeof activationMap;


