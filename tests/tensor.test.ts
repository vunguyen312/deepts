import { test } from "node:test";
import assert from "node:assert/strict";
import { Tensor } from "../src/math/Tensor";

test("Tensor is created with correct data, shape, and strides", () => {
    const tensor = new Tensor([[1, 2], [3, 4]]);
    assert.deepEqual(tensor.data, new Float32Array([1, 2, 3, 4]));
    assert.deepEqual(tensor.shape, [2, 2]);
    assert.deepEqual(tensor.strides, [2, 1]);
});

test("Tensor flattens 1D nested arrays", () => {
    const tensor = new Tensor([1, 2, 3]);
    assert.deepEqual(tensor.data, new Float32Array([1, 2, 3]));
    assert.deepEqual(tensor.shape, [3]);
    assert.deepEqual(tensor.strides, [1]);
});

test("Tensor accepts a scalar", () => {
    const tensor = new Tensor(5);
    assert.deepEqual(tensor.shape, []);
    assert.deepEqual(tensor.data, new Float32Array([5]));
    assert.deepEqual(tensor.strides, []);
});

test("Tensor typed overload accepts matching shape", () => {
    const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    assert.deepEqual(tensor.strides, [3, 1]);
});

test("Tensor rejects mismatched element count", () => {
    assert.throws(
        () => new Tensor(new Float32Array([1, 2, 3]), [4]),
        /requires/
    );
});

test("Tensor rejects zero-length dimensions", () => {
    assert.throws(
        () => new Tensor(new Float32Array(0), [0]),
        /invalid dimension/
    );
});

test("Tensor rejects negative dimensions", () => {
    assert.throws(
        () => new Tensor(new Float32Array([1, 2, 3]), [-1, -3]),
        /invalid dimension/
    );
});

test("Tensor rejects fractional dimensions", () => {
    assert.throws(
        () => new Tensor(new Float32Array([1, 2, 3]), [1.5, 2]),
        /invalid dimension/
    );
});

test("Tensor rejects ragged nested arrays", () => {
    assert.throws(() => new Tensor([[1, 2], [3]]), /mismatched/);
});

test("Tensor rejects empty subarrays", () => {
    assert.throws(() => new Tensor([[], []]), /empty/);
    assert.throws(() => new Tensor([]), /empty/);
});

test("Tensor rejects mixed scalar/array nesting", () => {
    assert.throws(() => new Tensor([1, [2, 3]]), /mismatched data types/);
});

test("Tensor.rand returns a Tensor with the requested shape", () => {
    const tensor = Tensor.rand(2, 3);
    assert.deepEqual(tensor.shape, [2, 3]);
});

test("Tensor.rand fills data with one float32 per element", () => {
    const tensor = Tensor.rand(2, 3, 4);
    assert.ok(tensor.data instanceof Float32Array);
    assert.equal(tensor.data.length, 2 * 3 * 4);
});

test("Tensor.rand elements are finite and within float32 [0, 1]", () => {
    const tensor = Tensor.rand(10, 10);
    for (const element of tensor.data) {
        assert.ok(Number.isFinite(element), `value ${element} is not finite`);
        assert.ok(element >= 0 && element <= 1, 
                  `value ${element} out of range [0, 1]`);
    }
});
test("Tensor.xavier returns a Tensor with the requested shape", () => {
    const tensor = Tensor.xavier(2, 3, 2, 3);
    assert.deepEqual(tensor.shape, [2, 3]);
    assert.deepEqual(tensor.strides, [3, 1]);
});

test("Tensor.xavier bounds values by the fan-in/fan-out limit", () => {
    const limit = Math.sqrt(6 / (2 + 3));
    const tensor = Tensor.xavier(2, 3, 2, 3);
    for (const element of tensor.data) {
        assert.ok(Number.isFinite(element), `value ${element} is not finite`);
        assert.ok(element >= -limit && element <= limit,
                  `value ${element} out of range [-${limit}, ${limit}]`);
    }
});

test("Tensor.rand computes row-major strides", () => {
    const tensor = Tensor.rand(2, 3, 4);
    assert.deepEqual(tensor.strides, [12, 4, 1]);
});

test("Tensor.rand draws differ across calls", () => {
    const tensorA = Tensor.rand(100).data;
    const tensorB = Tensor.rand(100).data;
    let sawDifference = false;
    for (let i = 0; i < tensorA.length; i++) {
        if (tensorA[i] !== tensorB[i]) {
            sawDifference = true;
            break;
        }
    }
    assert.ok(sawDifference, "two rand calls produced identical data");
});

test("Tensor.zeros returns a Tensor with the requested shape", () => {
    const tensor = Tensor.zeros(2, 3);
    assert.deepEqual(tensor.shape, [2, 3]);
});

test("Tensor.zeros fills data with one float32 per element", () => {
    const tensor = Tensor.zeros(2, 3, 4);
    assert.ok(tensor.data instanceof Float32Array);
    assert.equal(tensor.data.length, 2 * 3 * 4);
});

test("Tensor.zeros elements are zero", () => {
    const tensor = Tensor.zeros(10, 10);
    for (const element of tensor.data) {
        assert.ok(element === 0, `value ${element} is non-zero`);
    }
});

test("Tensor.zero sets all elements to zero inplace", () => {
    const tensor = Tensor.rand(10, 10);
    tensor.zero();
    for (const element of tensor.data) {
        assert.ok(element === 0, `value ${element} is non-zero`);
    }
});

test("Tensor.dot produces the dot product of two vectors", () => {
    const vec1 = new Tensor([1, 2, 3]);
    const vec2 = new Tensor([4, 5, 6]);
    const dotResult = vec1.dot(vec2);
    assert.deepEqual(dotResult, new Tensor(1 * 4 + 2 * 5 + 3 * 6));
});

test("Tensor.dot rejects non-vector operands", () => {
    assert.throws(
        () => {
            const vec1 = new Tensor([1, 2, 3]);
            const vec2 = new Tensor([[1, 2], [3, 4]]);
            vec1.dot(vec2);
        },
        /must be/
    );
});

test("Tensor.dot rejects vectors of mismatched lengths", () => {
    assert.throws(
        () => {
            const vec1 = new Tensor([1, 2, 3]);
            const vec2 = new Tensor([4, 5, 6, 7]);
            vec1.dot(vec2);
        },
        /same shape/
    );
});

test("Tensor.matmul produces the matrix multiplication of two matrices", () => {
    const mat1 = new Tensor([[-1, 4], [2, 3]]);
    const mat2 = new Tensor([[9, -3], [6, 1]]);
    const mulResult = mat1.matmul(mat2);
    assert.deepEqual(mulResult.data, new Float32Array([15, 7, 36, -3]));
    assert.deepEqual(mulResult.shape, [2, 2]);
    assert.deepEqual(mulResult.strides, [2, 1]);
});

test("Tensor.matmul multiplies a matrix by a column vector", () => {
    const mat = new Tensor([[-1, 4], [2, 3]]);
    const vec = new Tensor([2, -1]);
    const result = mat.matmul(vec);
    assert.deepEqual(result.data, new Float32Array([-6, 1]));
    assert.deepEqual(result.shape, [2]);
    assert.deepEqual(result.strides, [1]);
});

test("Tensor.matmul multiplies a row vector by a matrix", () => {
    const vec = new Tensor([2, -1]);
    const mat = new Tensor([[-1, 4], [2, 3]]);
    const result = vec.matmul(mat);
    assert.deepEqual(result.data, new Float32Array([-4, 5]));
    assert.deepEqual(result.shape, [2]);
});

test("Tensor.matmul multiplies a non-square matrix by a column vector", () => {
    const mat = new Tensor([[1, 2], [3, 4], [5, 6]]);
    const vec = new Tensor([7, 8]);
    const result = mat.matmul(vec);
    assert.deepEqual(result.data, new Float32Array([23, 53, 83]));
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(result.strides, [1]);
});

test("Tensor.matmul multiplies a row vector by a non-square matrix", () => {
    const vec = new Tensor([1, 2]);
    const mat = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const result = vec.matmul(mat);
    assert.deepEqual(result.data, new Float32Array([9, 12, 15]));
    assert.deepEqual(result.shape, [3]);
});

test("Tensor.matmul rejects vectors with incompatible dimensions", () => {
    assert.throws(
        () => {
            const vec = new Tensor([1, 2, 3]);
            const mat = new Tensor([[1, 2], [3, 4]]);
            vec.matmul(mat);
        },
        /incompatibility/
    );
});

test("Tensor.matmul rejects operands with more than two dimensions", () => {
    assert.throws(
        () => {
            const cube = new Tensor([[[1]]]);
            const mat = new Tensor([[1]]);
            cube.matmul(mat);
        },
        /1D or 2D/
    );
});

test("Tensor.matmul rejects matrices of incompatible dimensions", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1]]);
            const mat2 = new Tensor([[1, 2], [3, 4]]);
            mat1.matmul(mat2);
        },
        /incompatibility/
    );
});

test("Tensor.muls multiples element-wise in place", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    mat1.muls(mat2);
    assert.deepEqual(mat1.data, new Float32Array([2, 4, 6, 8, 10, 12]));
});

test("Tensor.muls rejects tensors of incompatible dimensions", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
            const mat2 = new Tensor([[2, 2], [2, 2], [2, 2]]);
            mat1.muls(mat2);
        },
        /broadcast:/
    )
});

test("Tensor.mul multiples element-wise and returns a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    const result = mat1.mul(mat2);
    assert.deepEqual(mat1.data, new Float32Array([1, 2, 3, 4, 5, 6]));
    assert.deepEqual(result.data, new Float32Array([2, 4, 6, 8, 10, 12]));
});

test("Tensor.adds adds element-wise in place", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    mat1.adds(mat2);
    assert.deepEqual(mat1.data, new Float32Array([3, 4, 5, 6, 7, 8]));
});

test("Tensor.adds rejects tensors of incompatible dimensions", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
            const mat2 = new Tensor([[2, 2], [2, 2], [2, 2]]);
            mat1.adds(mat2);
        },
        /broadcast:/
    )
});

test("Tensor.add adds element-wise and returns a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    const result = mat1.add(mat2);
    assert.deepEqual(mat1.data, new Float32Array([1, 2, 3, 4, 5, 6]));
    assert.deepEqual(result.data, new Float32Array([3, 4, 5, 6, 7, 8]));
});

test("Tensor.subs subtracts element-wise in place", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    mat1.subs(mat2);
    assert.deepEqual(mat1.data, new Float32Array([-1, 0, 1, 2, 3, 4]));
});

test("Tensor.subs rejects tensors of incompatible dimensions", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
            const mat2 = new Tensor([[2, 2], [2, 2], [2, 2]]);
            mat1.subs(mat2);
        },
        /broadcast:/
    )
});

test("Tensor.sub subtracts element-wise and returns a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor ([[2, 2, 2], [2, 2, 2]]);
    const result = mat1.sub(mat2);
    assert.deepEqual(mat1.data, new Float32Array([1, 2, 3, 4, 5, 6]));
    assert.deepEqual(result.data, new Float32Array([-1, 0, 1, 2, 3, 4]));
});

test("Tensor.scales multiples element-wise in place by a scalar", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    mat1.scales(2);
    assert.deepEqual(mat1.data, new Float32Array([2, 4, 6, 8, 10, 12]));
});

test("Tensor.scale multiples by a factor, returning a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const result = mat1.scale(2);
    assert.deepEqual(mat1.data, new Float32Array([1, 2, 3, 4, 5, 6]));
    assert.deepEqual(result.data, new Float32Array([2, 4, 6, 8, 10, 12]));
});

test("Tensor.maps performs a callback on every element in place", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    mat1.maps(element => element + 1);
    assert.deepEqual(mat1.data, new Float32Array([2, 3, 4, 5, 6, 7]));
});

test("Tensor.map performs a callback on every element on a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const result = mat1.map(element => element + 1);
    assert.deepEqual(mat1.data, new Float32Array([1, 2, 3, 4, 5, 6]));
    assert.deepEqual(result.data, new Float32Array([2, 3, 4, 5, 6, 7]));
});

test("Tensor.transposes transposes a tensor in place", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    mat1.transposes();
    assert.deepEqual(mat1.shape, [3, 2]);
    assert.deepEqual(mat1.strides, [1, 3]);
});

test("Tensor.transposes rejects tensors due to only one arg passed", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
            mat1.transposes(0);
        },
        /provide two/
    )
});

test("Tensor.transposes rejects tensors due to out of bounds args", () => {
    assert.throws(
        () => {
            const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
            mat1.transposes(-1, 0);
        },
        /Expected/
    )
});

test("Tensor.transposes transposes a given tensor returns a new tensor", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const result = mat1.transpose();
    assert.deepEqual(mat1.shape, [2, 3]);
    assert.deepEqual(mat1.strides, [3, 1]);
    assert.deepEqual(result.shape, [3, 2]);
    assert.deepEqual(result.strides, [1, 3]);
});

test("Tensor.matmul multiplies a transposed matrix by a matrix", () => {
    const mat1 = new Tensor([[-1, 2], [4, 3]]);
    const mat2 = new Tensor([[9, -3], [6, 1]]);
    mat1.transposes();
    const result = mat1.matmul(mat2);
    assert.deepEqual(result.data, new Float32Array([15, 7, 36, -3]));
    assert.deepEqual(result.shape, [2, 2]);
    assert.deepEqual(result.strides, [2, 1]);
});

test("Tensor.matmul multiplies a matrix by a transposed matrix", () => {
    const mat1 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const mat2 = new Tensor([[7, 8, 9], [10, 11, 12]]);
    mat2.transposes();
    const result = mat1.matmul(mat2);
    assert.deepEqual(result.data, new Float32Array([50, 68, 122, 167]));
    assert.deepEqual(result.shape, [2, 2]);
    assert.deepEqual(result.strides, [2, 1]);
});

test("Tensor.matmul multiplies a transposed matrix by a column vector", () => {
    const mat1 = new Tensor([[1, 2], [3, 4]]);
    mat1.transposes();
    const vec = new Tensor([5, 6]);
    const result = mat1.matmul(vec);
    assert.deepEqual(result.data, new Float32Array([23, 34]));
    assert.deepEqual(result.shape, [2]);
    assert.deepEqual(result.strides, [1]);
});

test("Tensor.matmul multiplies a row vector by a transposed matrix", () => {
    const vec = new Tensor([5, 6]);
    const mat = new Tensor([[1, 2], [3, 4]]);
    mat.transposes();
    const result = vec.matmul(mat);
    assert.deepEqual(result.data, new Float32Array([17, 39]));
    assert.deepEqual(result.shape, [2]);
    assert.deepEqual(result.strides, [1]);
});

test("Tensor.reshapes reassigns a Tensor's shape array", () => {
    const tensor1 = new Tensor([[[1, 2], [1, 2]], [[4, 2], [5, 2]]]);
    tensor1.reshapes(8);
    assert.deepEqual(tensor1.shape, [8]);
    assert.deepEqual(tensor1.strides, [1]);
});

test("Tensor.reshapes reassigns a Tensor's shape array 2", () => {
    const tensor1 = new Tensor([[1, 3, 4], [2, 3, 5]]);
    tensor1.reshapes(2, 3);
    assert.deepEqual(tensor1.shape, [2, 3]);
    assert.deepEqual(tensor1.strides, [3, 1]);
});

test("Tensor.reshapes reassigns a Tensor's shape array 2", () => {
    const tensor1 = new Tensor([[[[1]]]]);
    tensor1.reshapes(1);
    assert.deepEqual(tensor1.shape, [1]);
    assert.deepEqual(tensor1.strides, [1]);
});


test("Tensor.reshapes rejects invalid shapes for a given Tensor", () => {
    assert.throws(
        () => {
            const tensor1 = new Tensor([[[1, 2], [1, 2]], [[4, 2], [5, 2]]]);
            tensor1.reshapes(7);
        },
        /requires 7/
    )
});

test("Tensor.reshape creates new Tensor with given data and shape", () => {
    const tensor1 = new Tensor([[[1, 2], [1, 2]], [[4, 2], [5, 2]]]);
    const result = tensor1.reshape(8);
    assert.deepEqual(tensor1.shape, [2, 2, 2]);
    assert.deepEqual(tensor1.strides, [4, 2, 1]);
    assert.deepEqual(result.shape, [8]);
    assert.deepEqual(result.strides, [1]);
});

test("Tensor.adds can add a transposed matrix of same shape", () => {
    const mat1 = new Tensor([[1, 2], [3, 4], [5, 6]]);
    const mat2 = new Tensor([[1, 2, 3], [4, 5, 6]]);
    mat2.transposes();
    mat1.adds(mat2);
    assert.deepEqual(mat1.data, new Float32Array([2, 6, 5, 9, 8, 12]));
});

test("Tensor.broadcast properly expands two Tensors to add", () => {
    const tensor1 = new Tensor([1, 2, 3]);
    const tensor2 = new Tensor([1]);
    tensor1.adds(tensor2);
    assert.deepEqual(tensor1.data, new Float32Array([2, 3, 4]));
});

test("Tensor.broadcast properly expands a scalar to add", () => {
    const tensor1 = new Tensor([1, 2, 3]);
    const tensor2 = new Tensor(4);
    tensor1.adds(tensor2);
    assert.deepEqual(tensor1.data, new Float32Array([5, 6, 7]));
});

test("Tensor.broadcast fails to expand incompatible dimensions", () => {
    assert.throws(
        () => {
            const tensor1 = new Tensor([1, 2, 3]);
            const tensor2 = new Tensor([1, 2]);
            tensor1.adds(tensor2);
        },
        /broadcast:/
    );
});

test("Tensor.muls broadcasts a rank-shorter unit tensor over all elements", () => {
    const tensor = new Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]);
    tensor.muls(new Tensor([2]));
    assert.deepEqual(
        tensor.data,
        new Float32Array([2, 4, 6, 8, 10, 12, 14, 16])
    );
});

test("Tensor.adds broadcasts a bias row over batch rows", () => {
    const tensor = new Tensor([[1, 2], [3, 4]]);
    tensor.adds(new Tensor([10, 20]));
    assert.deepEqual(tensor.data, new Float32Array([11, 22, 13, 24]));
});

test("Tensor.muls broadcasts matching shapes that share no extent", () => {
    const tensor = new Tensor([[1, 2, 3, 4]]);
    tensor.muls(new Tensor([2, 4, 6, 8]));
    assert.deepEqual(tensor.data, new Float32Array([2, 8, 18, 32]));
});

test("Tensor.mul broadcasts to a larger result shape", () => {
    const left = new Tensor([[1], [2]]);
    const right = new Tensor([10, 20, 30]);
    const result = left.mul(right);
    assert.deepEqual(result.shape, [2, 3]);
    assert.deepEqual(
        result.data,
        new Float32Array([10, 20, 30, 20, 40, 60])
    );

    const reversed = new Tensor([1, 2, 3]).add(new Tensor([[10], [20]]));
    assert.deepEqual(reversed.shape, [2, 3]);
    assert.deepEqual(reversed.data, new Float32Array([11, 12, 13, 21, 22, 23]));
});

test("Tensor.mul broadcasts over higher-rank operands", () => {
    const tensorA = new Tensor([[[1], [2]]]);
    const tensorB = new Tensor([[[1, 2, 3]], [[4, 5, 6]]]);
    const result = tensorA.mul(tensorB);
    assert.deepEqual(result.shape, [2, 2, 3]);
    assert.deepEqual(
        result.data,
        new Float32Array([1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12])
    );
});

test("Tensor.adds rejects in-place broadcasts that would expand the target", () => {
    const tensor = new Tensor([[1], [2], [3]]);
    assert.throws(
        () => tensor.adds(new Tensor([10, 20, 30])),
        /same shape/
    );
    assert.throws(
        () => tensor.muls(new Tensor([[10, 20], [30, 40]])),
        /broadcast:/
    );
});