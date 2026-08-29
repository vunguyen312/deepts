// Waterloo MATH136 trauma #thosewhoknow

type NestedNumberArray = number | NestedNumberArray[];
type Transformer<T, R> = (input: T) => R;

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

const assertDims = (tensor: Tensor, op: string, 
                    expectedDims: number): void => {
    const { length } = tensor.shape;
    if (length === expectedDims) {
        return;
    }
    throw new Error(
        `${op}: Tensors must be ${expectedDims}D, given ${length}.`
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
    private _shape: number[];
    private _strides: number[];

    constructor(data: NestedNumberArray);
    constructor(data: Float32Array, shape: number[]);

    constructor(data: NestedNumberArray | Float32Array, shape?: number[]) {
        if (data instanceof Float32Array) {
            this.data = new Float32Array(data);
            this._shape = [...shape!];
            assertValidShape(data.length, shape!);
            this._strides = this.calcStrides();
            return;
        }

        const flatData: number[] = [];
        this._shape = [];
        const layer = 0;
        this.walk(data, flatData, layer);
        this._strides = this.calcStrides();
        this.data = new Float32Array(flatData);
    }

    private calcStrides(): number[] {
        const strides = new Array(this._shape.length).fill(1);
        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this._shape[i + 1];
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

        const currDim = this._shape[layer];
        if (!currDim) {
            this._shape[layer] = data.length;
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
    
    public static rand(...shape: number[]): Tensor {
        const totalElements = shape.reduce((acc, curr) => acc * curr, 1);
        const data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
            data[i] = Math.random();
        }
        return new Tensor(data, shape);
    }

    public static xavier(fanIn: number, fanOut: number, 
                         ...shape: number[]): Tensor {
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
    
    public static zeros(...shape: number[]): Tensor {
        const totalElements = shape.reduce((acc, curr) => acc * curr, 1);
        const data = new Float32Array(totalElements);
        return new Tensor(data, shape);
    }

    private static broadcast(tensor1: Tensor, tensor2: Tensor): number[] {
        /* Very tired for today so I will note how broadcasting works.
           Libs like PyTorch DO NOT create another Tensor in memory to broadcast.
           They instead virtualize expansion by setting the strides of expanding dimensions to 0.
           For two tensors to be broadcastable, they must be EITHER: Equal in shapes, missing one dimensions
           compared to the other, or has dimensions of size 1. Take for example the tensors (3, 3) and (3, 1)
           We modify the second one's stride, the Y stride to 0. This will also require refactoring adds, muls,
           and subs
           */
        let largerTensor = tensor1;
        let smallerTensor = tensor2;
        if (tensor1.shape.length < tensor2.shape.length) {
            largerTensor = tensor2;
            smallerTensor = tensor1;
        }
        const largerTensorLen = largerTensor.shape.length;
        const smallerTensorLen = smallerTensor.shape.length;
        const dimensionsToFill = largerTensorLen - smallerTensorLen;
        const UNIT_DIMS = 1;
        const newShape = new Array(dimensionsToFill).fill(UNIT_DIMS);

        return [1, 2];
    }

    public dot(tensor: Tensor): Tensor {
        const VECTOR_DIMS = 1;
        assertDims(this, "dot", VECTOR_DIMS);
        assertSameShape(this, tensor);

        let result = new Float32Array(1);
        for (let i = 0; i < this.data.length; i++) {
            result[0] += this.data[i] * tensor.data[i];
        }
        return new Tensor(result, []);
    }

    public matmul(tensor: Tensor): Tensor {
        const MATRIX_DIMS = 2;
        const SCALAR_DIMS = 0;
        const leftDims = this._shape.length;
        const rightDims = tensor.shape.length;
        if (leftDims > MATRIX_DIMS || rightDims > MATRIX_DIMS
            || leftDims === SCALAR_DIMS || rightDims === SCALAR_DIMS
        ) {
            throw new Error(
                `matmul: Tensors must be 1D or 2D, given ` +
                `${leftDims}D and ${rightDims}D.`
            );
        }
        
        const VECTOR_DIMS = 1;
        const UNIT_DIMS = 1;
        let leftShape = [...this._shape];
        let rightShape = [...tensor.shape];
        if (leftDims === VECTOR_DIMS) {
            leftShape = [UNIT_DIMS, this._shape[0]];
        }
        if (rightDims === VECTOR_DIMS) {
            rightShape = [tensor.shape[0], UNIT_DIMS];
        }

        const leftRowCount = leftShape[0];
        const leftColCount = leftShape[1];
        const rightRowCount = rightShape[0];
        const rightColCount = rightShape[1];
        if (leftColCount !== rightRowCount) {
            throw new Error(
                `matmul: Matrix shape incompatibility, given ` +
                `${this._shape} and ${tensor.shape}.`
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

        const resultShape = leftDims === VECTOR_DIMS 
                            || rightDims === VECTOR_DIMS
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
        const result = new Tensor(this.data, this._shape);
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
        const result = new Tensor(this.data, this._shape);
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
        const result = new Tensor(this.data, this._shape);
        result.subs(tensor);
        return result;
    }

    public scales(factor: number): void {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] *= factor;
        }
    }

    public scale(factor: number): Tensor {
        const result = new Tensor(this.data, this._shape);
        result.scales(factor);
        return result;
    }
    
    public maps(callback: Transformer<number, number>): void {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = callback(this.data[i]);
        }
    }

    public map(callback: Transformer<number, number>): Tensor {
        const result = new Tensor(this.data, this._shape);
        result.maps(callback);
        return result;
    }

    public transposes(dim1?: number, dim2?: number): void {
        const MATRIX_DIMS = 2;
        if (this._shape.length < MATRIX_DIMS) {
            throw new Error("transpose: Cannot transpose below 2D");
        }

        if (dim1 === undefined && dim2 === undefined
            && this._shape.length === MATRIX_DIMS) {
            this.transposes(0, 1);
            return;
        }

        if (dim1 === undefined || dim2 === undefined) {
            throw new Error("transpose: Must provide two dimensions to swap");
        }

        const MIN_INDEX = 0;
        const MAX_INDEX = this._shape.length - 1;
        if (dim1 < MIN_INDEX || dim2 < MIN_INDEX 
            || dim1 > MAX_INDEX || dim2 > MAX_INDEX) {
            throw new Error(
                `transpose: Expected indices ` +
                `0 <= x <= ${MAX_INDEX}, ` +
                `given ${dim1} and ${dim2}` 
            );
        }

        const oldShape = [...this._shape];
        const oldStrides = [...this.strides];
        this._shape[dim1] = oldShape[dim2];
        this._shape[dim2] = oldShape[dim1];
        const newStrides = this.calcStrides();
        const newData = new Float32Array(this.data.length);
        for (let i = 0; i < newData.length; i++) {
            let remaining = i;
            let oldOffset = 0;
            for (let j = 0; j < this._shape.length; j++) {
                const coord = Math.floor(remaining / newStrides[j]);
                remaining %= newStrides[j];
                let oldDim = j;
                if (j === dim1) {
                    oldDim = dim2;
                }
                if (j === dim2) {
                    oldDim = dim1;
                }
                oldOffset += coord * oldStrides[oldDim];
            }
            newData[i] = this.data[oldOffset];
        }
    
    this.data = newData;
    this._strides = newStrides;
    }

    public transpose(): Tensor {
        const result = new Tensor(this.data, this._shape);
        result.transposes();
        return result;
    }

    public reshapes(...shape: number[]): void {
        assertValidShape(this.data.length, shape);
        this._shape = [...shape];
        this._strides = this.calcStrides();
    }

    public reshape(...shape: number[]): Tensor {
        const result = new Tensor(this.data, shape);
        return result;
    }

    public get shape(): number[] {
        return [...this._shape];
    }

    public get strides(): number[] {
        return [...this._strides];
    }
}