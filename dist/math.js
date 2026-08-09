"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Matrix = exports.Vector = void 0;
class Vector {
    static randomVector(fanIn, fanOut) {
        const a = Math.sqrt(6 / (fanIn + fanOut));
        const result = [];
        for (let i = 0; i < fanIn; i++) {
            result.push(Math.random() * 2 * a - a);
        }
        return result;
    }
    static dot(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors are not of the same length');
        }
        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }
        return sum;
    }
    static scalarMul(scalar, vec) {
        const result = [...vec];
        for (let i = 0; i < vec.length; i++) {
            result[i] *= scalar;
        }
        return result;
    }
    static add(vec1, vec2) {
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
exports.Vector = Vector;
class Matrix {
    static zeroMat(rowLength, colLength) {
        const result = [];
        for (let i = 0; i < rowLength; i++) {
            const col = new Array(colLength).fill(0);
            result.push(col);
        }
        return result;
    }
    static mul(mat1, mat2) {
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
    static getRow(mat, j) {
        const result = [];
        for (let i = 0; i < mat.length; i++) {
            result.push(mat[i][j]);
        }
        return result;
    }
}
exports.Matrix = Matrix;
//# sourceMappingURL=math.js.map