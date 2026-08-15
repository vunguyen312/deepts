// Waterloo MATH136 trauma #thosewhoknow

export class Vector {
    public static randomVector(fanIn: number, fanOut: number): Float32Array {
        const a = Math.sqrt(6 / (fanIn + fanOut));
        const result: number[] = [];
        for (let i = 0; i < fanIn; i++) {
            result.push(Math.random() * 2 * a - a);
        }
        return new Float32Array(result);
    }

    public static dot(vec1: Float32Array, vec2: Float32Array): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    public static scalarMul(scalar: number, vec: Float32Array): Float32Array {
        const result = new Float32Array(vec);
        for (let i = 0; i < vec.length; i++) {
            result[i] *= scalar;
        }

        return result;
    }

    public static add(vec1: Float32Array, vec2: Float32Array): Float32Array {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        const result = new Float32Array(vec1);
        for (let i = 0; i < vec1.length; i++) {
            result[i] += vec2[i];
        }

        return result;
    }
}

export class Matrix {
    public static zeroMat(rowLength: number, colLength: number): Float32Array[] {
        const result: Float32Array[] = [];
        for (let i = 0; i < rowLength; i++) {
            const col = new Float32Array(colLength);
            result.push(col);
        }

        return result;
    }

    public static mul(mat1: Float32Array[], mat2: Float32Array[]): Float32Array[] {
        const rowLength = mat2.length;
        const colLength = mat1[0].length;
        const result = Matrix.zeroMat(rowLength, colLength);
        for (let i = 0; i < colLength; i++) {
            const jthRow = Matrix.getRow(mat1, i);
            for (let j = 0; j < rowLength; j++) {
                const ithCol = mat2[j];
                result[j][i] = Vector.dot(jthRow, ithCol);
            }
        }

        return result;
    }

    public static getRow(mat: Float32Array[], j: number): Float32Array {
        const result: number[] = [];
        for (let i = 0; i < mat.length; i++) {
            result.push(mat[i][j]);
        }

        return new Float32Array(result);
    }
}