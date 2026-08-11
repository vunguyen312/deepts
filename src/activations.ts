export interface Activation {
    readonly fn: (x: number) => number;
    readonly derivative: (x: number) => number;
}

export const step: Activation = {
    fn: x => (x > 0 ? 1 : 0),
    derivative: _x => 0
};

export const relu: Activation = {
    fn: x => Math.max(0, x),
    derivative: x => (x > 0 ? 1 : 0),
};

export const sigmoid: Activation = {
    fn: x => 1 / (1 + Math.exp(-x)),
    derivative: x => {
        const sigmoidValue = sigmoid.fn(x);
        return sigmoidValue * (1 - sigmoidValue);
    }
};

export const activationMap = {
    step,
    relu,
    sigmoid
} satisfies Record<string, Activation>;
// Builds a mapping for Activations by their strings
export type ActivationFunc = keyof typeof activationMap;


