export class Vector {
    public static randomVector(fanIn: number, fanOut: number): number[] {
        const a = Math.sqrt(6 / (fanIn + fanOut));
        const result: number[] = [];
        for (let i = 0; i < fanIn; i++) {
            result.push(Math.random() * 2 * a - a);
        }
        return result;
    }

    public static dot(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
    }

    public static scalarMul(scalar: number, vec: number[]): number[] {
        const result = [...vec];
        for (let i = 0; i < vec.length; i++) {
            result[i] *= scalar;
        }

        return result;
    }

    public static add(vec1: number[], vec2: number[]): number[] {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }

        const result = [...vec1];
        for (let i = 0; i < vec1.length; i++) {
            result[i] += vec2[i];
        }

        return result;
    }
}

export class Matrix {
    public static zeroMat(rowLength: number, colLength: number): number[][] {
        const result: number[][] = [];
        for (let i = 0; i < rowLength; i++) {
            const col = new Array(colLength).fill(0);
            result.push(col);
        }

        return result;
    }

    public static mul(mat1: number[][], mat2: number[][]): number[][] {
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

    public static getRow(mat: number[][], j: number): number[] {
        const result: number[] = [];
        for (let i = 0; i < mat.length; i++) {
            result.push(mat[i][j]);
        }

        return result;
    }
}