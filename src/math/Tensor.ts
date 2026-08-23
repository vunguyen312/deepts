type NestedNumberArray = number | NestedNumberArray[];
type TensorOperation = "dot" | "matmul";

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
            this.assertValidShape();
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

    private assertValidShape(): void {
        let totalElements = 1;
        for (let i = 0; i < this.shape.length; i++) {
            const currDim = this.shape[i];
            const isCurrDimSafeInt = Number.isSafeInteger(currDim);
            if (!isCurrDimSafeInt || currDim <= 0) {
                throw new Error(
                    `Tensor contains invalid dimension ${currDim}. Must be ` +
                    `positive safe integer.`
                );
            }
            totalElements *= this.shape[i];
        }
        if (totalElements !== this.data.length) {
            throw new Error(
                `Tensor of dimensions [${this.shape}] requires ` +
                `${totalElements} elements but got ${this.data.length}.`
            );
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

    private static assertDims(tensor: Tensor, op: TensorOperation,
                               expectedDims: number): void {
        if (tensor.shape.length === expectedDims) {
            return;
        }
        throw new Error(
            `${op}: Tensors must be ${expectedDims}D, given ` +
            `${tensor.shape.length}.`
        );
    }

    public dot(tensor: Tensor): number {
        const VECTOR_DIMS = 1;
        Tensor.assertDims(this, "dot", VECTOR_DIMS);
        Tensor.assertDims(tensor, "dot", VECTOR_DIMS);

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
        const MATRIX_DIMS = 2;
        Tensor.assertDims(this, "matmul", MATRIX_DIMS);
        Tensor.assertDims(tensor, "matmul", MATRIX_DIMS);

        const matOneRowCount = this.shape[0];
        const matTwoRowCount = tensor.shape[0];
        const matOneColCount = this.shape[1];
        const matTwoColCount = tensor.shape[1];
        if (matOneColCount !== matTwoRowCount) {
            throw new Error(
                `matmul: Matrix shape incompatibility, given ` +
                `${this.shape} and ${tensor.shape}.`
            );
        }

        const resultRowCount = matOneRowCount;
        const resultColCount = matTwoColCount;
        const resultData = new Float32Array(resultRowCount * resultColCount);
        for (let i = 0; i < resultRowCount; i++) {
            for (let j = 0; j < resultColCount; j++) {
                let sum = 0;
                for (let k = 0; k < matOneColCount; k++) {
                    sum += this.data[i * matOneColCount + k] *
                           tensor.data[k * matTwoColCount + j];
                }
                resultData[i * resultColCount + j] = sum;
            }
        }
        
        return new Tensor(resultData, [resultRowCount, resultColCount]);
    }
}