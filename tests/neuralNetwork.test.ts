import { test } from "node:test";
import assert from "node:assert/strict";
import { Layer } from "../src/core/neuralNetwork";
import { Tensor } from "../src/math/Tensor";

test("Layer.backward accumulates the outer-product weight gradient", () => {
    const layer = new Layer(
        "relu", 2, 3,
        new Float32Array([1, 0, 0, 1, 1, 1]),
        new Float32Array([0.5, -1, 2])
    );
    layer.forward(new Tensor([1, 2]));
    layer.backward(new Tensor([0.5, -0.25, 0.75]));

    const { gradWeights, gradBiases } = layer.getParams;
    assert.deepEqual(gradWeights.shape, [3, 2]);
    assert.deepEqual(gradBiases.shape, [3]);
    assert.deepEqual(
        gradWeights.data,
        new Float32Array([0.5, 1, -0.25, -0.5, 0.75, 1.5])
    );
    assert.deepEqual(gradBiases.data, new Float32Array([0.5, -0.25, 0.75]));
});
