// Waterloo MATH136 trauma #thosewhoknow

type NestedNumberArray = number | NestedNumberArray[];
type TensorOperation = "dot" | "matmul";

const assertValidShape = (dataLength: number, shape: number[]): void => {
    let totalElements = 1;
    for (const dim of shape) {
        const currDim = dim;
        const isCurrDimSafeInt = Number.isSafeInteger(currDim);
        if (isCurrDimSafeInt && currDim > 0) {
            totalElements *= dim;
            continue;
        }
        throw new Error(
            `Tensor contains invalid dimension ${currDim}. Must be ` +
            `positive safe integer.`
        );
    }

    if (totalElements === dataLength) {
        return;
    }
    throw new Error(
        `Tensor of dimensions [${shape}] requires ` +
        `${totalElements} elements but got ${dataLength}.`
    );
}

const assertDims = (tensor: Tensor, op: TensorOperation, 
                    expectedDims: number): void => {
    if (tensor.shape.length === expectedDims) {
        return;
    }
    throw new Error(
        `${op}: Tensors must be ${expectedDims}D, given ` +
        `${tensor.shape.length}.`
    );
}

const assertSameShape = (tensor1: Tensor, tensor2: Tensor): void => {
    let matchesDims = true;
    for (let i = 0; i < tensor1.shape.length; i++) {
        if (tensor1.shape[i] !== tensor2.shape[i]) {
            matchesDims = false;
        }
    }

    if (tensor1.shape.length === tensor2.shape.length && matchesDims) {
        return;
    }
    throw new Error(
        `Tensors must be of the same shape, given ` + 
        `${tensor1.shape} and ${tensor2.shape}`
    );
}

export class Tensor {
    public data: Float32Array;
    public shape: number[];
    public strides: number[];

    constructor(data: NestedNumberArray);
    constructor(data: Float32Array, shape: number[]);

    constructor(data: NestedNumberArray | Float32Array, shape?: number[]) {
        if (data instanceof Float32Array) {
            this.data = new Float32Array(data);
            this.shape = [...shape!];
            assertValidShape(data.length, shape!);
            this.strides = this.calcStrides();
            return;
        }

        const flatData: number[] = [];
        this.shape = [];
        const layer = 0;
        this.walk(data, flatData, layer);
        this.strides = this.calcStrides();
        this.data = new Float32Array(flatData);
    }

    private calcStrides(): number[] {
        const strides = new Array(this.shape.length).fill(1);
        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this.shape[i + 1];
        }
        return strides;
    }
    
    private walk(data: NestedNumberArray, flatData: number[], 
                 layer: number): void {
        if (typeof data === "number") {
            flatData.push(data);
            return;
        }

        if (data.length === 0) {
            throw new Error("Tensors cannot contain empty subarrays.");
        }

        const currDim = this.shape[layer];
        if (!currDim) {
            this.shape[layer] = data.length;
        } else if (currDim !== data.length) {
            throw new Error(
                `Tensor contains mismatched dimensions ${currDim} ` +
                `and ${data.length}.`
            );
        }

        let prevType = typeof data[0];
        for (let i = 0; i < data.length; i++) {
            const currType = typeof data[i];
            if (currType !== prevType) {
                throw new Error("Tensors cannot have mismatched data types.");
            }

            this.walk(data[i], flatData, layer + 1);
            prevType = currType;
        }
    }
    
    public static rand(shape: number[]): Tensor {
        const totalElements = shape.reduce((acc, curr) => acc * curr, 1);
        const data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
            data[i] = Math.random();
        }
        return new Tensor(data, shape);
    }

    public static xavier(shape: number[], fanIn: number,
                         fanOut: number): Tensor {
        const totalElements = shape.reduce((acc, curr) => acc * curr, 1);
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        const data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
            data[i] = Math.random() * 2 * limit - limit;
        }
        return new Tensor(data, shape);
    }

    public zero(): void {
        this.data.fill(0);
    }
    
    public static zeros(shape: number[]): Tensor {
        const totalElements = shape.reduce((acc, curr) => acc * curr, 1);
        const data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
            data[i] = 0;
        }
        return new Tensor(data, shape);
    }

    public dot(tensor: Tensor): number {
        const VECTOR_DIMS = 1;
        assertDims(this, "dot", VECTOR_DIMS);
        assertDims(tensor, "dot", VECTOR_DIMS);

        const vecOneLen = this.shape[0];
        const vecTwoLen = tensor.shape[0];
        if (vecOneLen !== vecTwoLen) {
            throw new Error(
                `dot: Vectors must be of the same dimension, given ` +
                `${vecOneLen} and ${vecTwoLen}.`
            );
        }

        let result = 0;
        for (let i = 0; i < vecOneLen; i++) {
            result += this.data[i] * tensor.data[i];
        }
        return result;
    }

    public matmul(tensor: Tensor): Tensor {
        // ok bro ill be honest this function is straight up robo-slop
        // just a placeholder until I figure out broadcasting
        const leftDims = this.shape.length;
        const rightDims = tensor.shape.length;
        if ((leftDims !== 1 && leftDims !== 2) ||
            (rightDims !== 1 && rightDims !== 2)) {
            throw new Error(
                `matmul: Tensors must be 1D or 2D, given ` +
                `${leftDims}D and ${rightDims}D.`
            );
        }
        if (leftDims === 1 && rightDims === 1) {
            throw new Error(
                `matmul: Vector-vector products must use dot, given ` +
                `${this.shape} and ${tensor.shape}.`
            );
        }

        const leftRowCount = leftDims === 1 ? 1 : this.shape[0];
        const leftColCount = leftDims === 1 ? this.shape[0] : this.shape[1];
        const rightRowCount = tensor.shape[0];
        const rightColCount = rightDims === 1 ? 1 : tensor.shape[1];
        if (leftColCount !== rightRowCount) {
            throw new Error(
                `matmul: Matrix shape incompatibility, given ` +
                `${this.shape} and ${tensor.shape}.`
            );
        }

        const resultRowCount = leftRowCount;
        const resultColCount = rightColCount;
        const resultData = new Float32Array(resultRowCount * resultColCount);
        for (let i = 0; i < resultRowCount; i++) {
            for (let j = 0; j < resultColCount; j++) {
                let sum = 0;
                for (let k = 0; k < leftColCount; k++) {
                    sum += this.data[i * leftColCount + k] *
                           tensor.data[k * rightColCount + j];
                }
                resultData[i * resultColCount + j] = sum;
            }
        }

        const resultShape = leftDims === 1 || rightDims === 1
            ? [resultRowCount * resultColCount]
            : [resultRowCount, resultColCount];
        return new Tensor(resultData, resultShape);
    }

    public muls(tensor: Tensor): void {
        assertSameShape(this, tensor);
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] *= tensor.data[i];
        }
    }

    public mul(tensor: Tensor): Tensor {
        const result = new Tensor(this.data, this.shape);
        result.muls(tensor);
        return result;
    }

    public adds(tensor: Tensor): void {
        assertSameShape(this, tensor);
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] += tensor.data[i];
        }
    }

    public add(tensor: Tensor): Tensor {
        const result = new Tensor(this.data, this.shape);
        result.adds(tensor);
        return result;
    }

    public subs(tensor: Tensor): void {
        assertSameShape(this, tensor);
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] -= tensor.data[i];
        }
    }

    public sub(tensor: Tensor): Tensor {
        const result = new Tensor(this.data, this.shape);
        result.subs(tensor);
        return result;
    }

    public scales(factor: number): void {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] *= factor;
        }
    }

    public scale(factor: number): Tensor {
        const result = new Tensor(this.data, this.shape);
        result.scales(factor);
        return result;
    }
}