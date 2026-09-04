// Waterloo MATH136 trauma #thosewhoknow

type NestedNumberArray = number | NestedNumberArray[];
type Transformer<T, R> = (input: T) => R;
type BinaryOp = (a: number, b: number) => number;

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

const assertSameShape = (a: Tensor | number[], b: Tensor | number[]): void => {
    let matchesDims = true;
    const shapeA = Array.isArray(a) ? a : a.shape;
    const shapeB = Array.isArray(b) ? b : b.shape;
    for (let i = 0; i < shapeA.length; i++) {
        if (shapeA[i] !== shapeB[i]) {
            matchesDims = false;
        }
    }

    if (shapeA.length === shapeB.length && matchesDims) {
        return;
    }
    throw new Error(
        `Tensors must be of the same shape, given ` + 
        `${shapeA} and ${shapeB}`
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

    private static broadcastShape(tensorA: Tensor, tensorB: Tensor): number[] {
        const UNIT_DIMS = 1;
        const { length: lengthA } = tensorA.shape;
        const { length: lengthB } = tensorB.shape;
        const newShapeLen = Math.max(lengthA, lengthB);
        const newShape = new Array(newShapeLen);

        const fillAEndIndex = newShapeLen - lengthA;
        const fillBEndIndex = newShapeLen - lengthB;
        const { shape: shapeA } = tensorA;
        const { shape: shapeB } = tensorB;
        for (let i = 0; i < newShapeLen; i++) {
            const currADim = shapeA[i - fillAEndIndex];
            const currBDim = shapeB[i - fillBEndIndex];
            const isUnitDim = currADim === UNIT_DIMS
                              || currBDim === UNIT_DIMS
                              || currADim === undefined
                              || currBDim === undefined;
            const isDimEqual = currADim === currBDim;
            if (isUnitDim || isDimEqual) {
                newShape[i] = Math.max(currADim || UNIT_DIMS, 
                                       currBDim || UNIT_DIMS);
                continue;
            }
            throw Error(
                `broadcast: Tensor shape incompatibility, given ` +
                `${shapeA} and ${shapeB}`
            );
        }
        
        return newShape;
    }

    private static broadcast(tensorA: Tensor, tensorB: Tensor): number[] {
        const UNIT_DIMS = 1;
        const { length: lengthA } = tensorA.strides;
        const { length: lengthB } = tensorB.strides;
        const newStridesLen = Math.max(lengthA, lengthB);
        const newStridesA = new Array(newStridesLen);
        
        const { shape: shapeA } = tensorA;
        const fillAEndIndex = newStridesLen - lengthA;
        for (let i = 0; i < newStridesA.length; i++) {
            const currDim = shapeA[i - fillAEndIndex];
            if (i < fillAEndIndex || currDim === UNIT_DIMS) {
                newStridesA[i] = 0;
                continue;
            }
            newStridesA[i] = tensorA.strides[i - fillAEndIndex];
        }

        return newStridesA;
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

    private isContiguous(shape: number[], strides: number[]): boolean {
        const lastStrideIndex = strides.length - 1;
        if (strides[lastStrideIndex] !== 1) {
            return false;
        }

        for (let i = lastStrideIndex - 1; i >= 0; i--) {
            const currStride = strides[i];
            const lastStride = strides[i + 1];
            const lastDim = shape[i + 1];
            if (currStride !== lastStride * lastDim) {
                return false;
            }
        }
        return true;
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
        const leftStrides = leftDims === VECTOR_DIMS
            ? [0, this._strides[0]]
            : this._strides;
        const rightStrides = rightDims === VECTOR_DIMS
            ? [tensor.strides[0], 0]
            : tensor.strides;
        for (let i = 0; i < resultRowCount; i++) {
            for (let j = 0; j < resultColCount; j++) {
                let sum = 0;
                for (let k = 0; k < leftColCount; k++) {
                    const leftOffset = i * leftStrides[0] + k * leftStrides[1];
                    const rightOffset = k * rightStrides[0] 
                        + j * rightStrides[1];
                    sum += this.data[leftOffset] * tensor.data[rightOffset];
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

    private static elementWiseWalk(dim: number, shape: number[],
                                   tensorA: Tensor, tensorB: Tensor,
                                   stridesA: number[], stridesB: number[],
                                   destData: Float32Array,
                                   destStrides: number[],
                                   offsetA: number, offsetB: number,
                                   offsetDest: number,
                                   op: BinaryOp): void {
        if (dim === shape.length) {
            destData[offsetDest] = op(tensorA.data[offsetA],
                                      tensorB.data[offsetB]);
            return;
        }

        for (let i = 0; i < shape[dim]; i++) {
            const nextDim = dim + 1;
            const newOffsetA = offsetA + i * stridesA[dim];
            const newOffsetB = offsetB + i * stridesB[dim];
            const newOffsetDest = offsetDest + i * destStrides[dim];
            Tensor.elementWiseWalk(nextDim, shape, tensorA, tensorB,
                                   stridesA, stridesB, destData, destStrides,
                                   newOffsetA, newOffsetB, newOffsetDest, op);
        }
    }

    private static elementWiseApply(tensorA: Tensor, tensorB: Tensor,
                                    op: BinaryOp): Tensor {
        const outputShape = Tensor.broadcastShape(tensorA, tensorB);
        const totalElements = outputShape.reduce((acc, curr) => acc * curr, 1);
        const resultData = new Float32Array(totalElements);
        const result = new Tensor(resultData, outputShape);
        const stridesA = Tensor.broadcast(tensorA, tensorB);
        const stridesB = Tensor.broadcast(tensorB, tensorA);

        const isAContiguous = tensorA.isContiguous(tensorA.shape, stridesA);
        const isBContiguous = tensorB.isContiguous(tensorB.shape, stridesB);
        if (isAContiguous && isBContiguous) {
            for (let i = 0; i < totalElements; i++) {
                result.data[i] = op(tensorA.data[i], tensorB.data[i]);
            }
            return result;
        }

        const FIRST_DIM_INDEX = 0;
        const INITIAL_OFFSET = 0;
        Tensor.elementWiseWalk(FIRST_DIM_INDEX, outputShape, tensorA, tensorB,
                               stridesA, stridesB, result.data,
                               result.strides, INITIAL_OFFSET, INITIAL_OFFSET,
                               INITIAL_OFFSET, op);
        return result;
    }

    private elementWiseInPlace(tensor: Tensor, op: BinaryOp): void {
        const outputShape = Tensor.broadcastShape(this, tensor);
        assertSameShape(outputShape, this._shape);

        const stridesA = Tensor.broadcast(this, tensor);
        const stridesB = Tensor.broadcast(tensor, this);

        const isAContiguous = this.isContiguous(this._shape, stridesA);
        const isBContiguous = this.isContiguous(tensor.shape, stridesB);
        if (isAContiguous && isBContiguous) {
            for (let i = 0; i < this.data.length; i++) {
                this.data[i] = op(this.data[i], tensor.data[i]);
            }
            return;
        }

        const FIRST_DIM_INDEX = 0;
        const INITIAL_OFFSET = 0;
        Tensor.elementWiseWalk(FIRST_DIM_INDEX, outputShape, this, tensor,
                               stridesA, stridesB, this.data, stridesA,
                               INITIAL_OFFSET, INITIAL_OFFSET,
                               INITIAL_OFFSET, op);
    }

    public muls(tensor: Tensor): void {
        this.elementWiseInPlace(tensor, (a, b) => a * b);
    }

    public mul(tensor: Tensor): Tensor {
        return Tensor.elementWiseApply(this, tensor, (a, b) => a * b);
    }

    public adds(tensor: Tensor): void {
        this.elementWiseInPlace(tensor, (a, b) => a + b);
    }

    public add(tensor: Tensor): Tensor {
        return Tensor.elementWiseApply(this, tensor, (a, b) => a + b);
    }

    public subs(tensor: Tensor): void {
        this.elementWiseInPlace(tensor, (a, b) => a - b);
    }

    public sub(tensor: Tensor): Tensor {
        return Tensor.elementWiseApply(this, tensor, (a, b) => a - b);
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

    public transposes(dimA?: number, dimB?: number): void {
        const MATRIX_DIMS = 2;
        if (this._shape.length < MATRIX_DIMS) {
            throw new Error("transpose: Cannot transpose below 2D");
        }

        if (dimA === undefined && dimB === undefined
            && this._shape.length === MATRIX_DIMS) {
            this.transposes(0, 1);
            return;
        }

        if (dimA === undefined || dimB === undefined) {
            throw new Error("transpose: Must provide two dimensions to swap");
        }

        const MIN_INDEX = 0;
        const MAX_INDEX = this._shape.length - 1;
        const isDim1ValidIndex = dimA >= MIN_INDEX && dimA <= MAX_INDEX;
        const isDim2ValidIndex = dimB >= MIN_INDEX && dimB <= MAX_INDEX;
        if (isDim1ValidIndex && isDim2ValidIndex) {
            const tempDim = this._shape[dimA];
            this._shape[dimA] = this._shape[dimB];
            this._shape[dimB] = tempDim;

            const tempStride = this._strides[dimA];
            this._strides[dimA] = this._strides[dimB];
            this._strides[dimB] = tempStride;
            return;
        }
        throw new Error(
            `transpose: Expected indices ` +
            `0 <= x <= ${MAX_INDEX}, ` +
            `given ${dimA} and ${dimB}` 
        );
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