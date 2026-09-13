import { test } from "node:test";
import assert from "node:assert/strict";
import { Softmax } from "../src/math/activations";
import { Tensor } from "../src/math/Tensor";

const sum = (data: Float32Array): number => {
    let total = 0;
    for (let i = 0; i < data.length; i++) {
        total += data[i];
    }
    return total;
};

const dot = (a: Float32Array, b: Float32Array): number => {
    let total = 0;
    for (let i = 0; i < a.length; i++) {
        total += a[i] * b[i];
    }
    return total;
};

const LOGITS = [0.5, -1.5, 2, 0.25, -0.75, 1.25];
const UPSTREAM = [0.3, -0.7, 0.9, 0.1, -0.2, 0.6];

const checkGradient = (shape: number[]): void => {
    const count = shape.reduce((acc, dim) => acc * dim, 1);
    const original = new Tensor(new Float32Array(LOGITS.slice(0, count)), 
                                shape);
    const upstream = new Tensor(new Float32Array(UPSTREAM.slice(0, count)), 
                                shape);
    const softmax = new Softmax();

    const EPSILON = 1e-3;
    const numeric = new Float64Array(count);
    for (let i = 0; i < count; i++) {
        const plus = new Tensor(original.data, shape);
        plus.data[i] += EPSILON;
        const minus = new Tensor(original.data, shape);
        minus.data[i] -= EPSILON;
        numeric[i] = (
            dot(softmax.forward(plus).data, upstream.data) -
            dot(softmax.forward(minus).data, upstream.data)
        ) / (2 * EPSILON);
    }

    softmax.forward(new Tensor(original.data, shape));
    const deltas = softmax.backward(upstream);
    for (let i = 0; i < count; i++) {
        assert.ok(
            Math.abs(deltas.data[i] - numeric[i]) < EPSILON,
            `shape [${shape}] index ${i}: backward ${deltas.data[i]} ` +
            `does not match numerical ${numeric[i]}`
        );
    }
};

test("Softmax backward matches the numerical Jacobian", () => {
    checkGradient([4]);
    checkGradient([2, 3]);
});

test("Softmax forward survives large logits", () => {
    const output = new Softmax().forward(new Tensor([1000, 1001, 1002]));
    for (const value of output.data) {
        assert.ok(Number.isFinite(value), `expected finite, got ${value}`);
    }
    assert.ok(Math.abs(sum(output.data) - 1) < 1e-6);
    assert.ok(Math.abs(output.data[1] / output.data[0] - Math.E) < 1e-3);
});

test("Softmax forward normalizes each row of a batch", () => {
    const output = new Softmax().forward(new Tensor([[1, 2], [3, 4]]));
    assert.ok(Math.abs(output.data[0] + output.data[1] - 1) < 1e-6);
    assert.ok(Math.abs(output.data[2] + output.data[3] - 1) < 1e-6);
});
