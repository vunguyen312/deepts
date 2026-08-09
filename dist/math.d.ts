export declare class Vector {
    static randomVector(fanIn: number, fanOut: number): number[];
    static dot(vec1: number[], vec2: number[]): number;
    static scalarMul(scalar: number, vec: number[]): number[];
    static add(vec1: number[], vec2: number[]): number[];
}
export declare class Matrix {
    static zeroMat(rowLength: number, colLength: number): number[][];
    static mul(mat1: number[][], mat2: number[][]): number[][];
    static getRow(mat: number[][], j: number): number[];
}
//# sourceMappingURL=math.d.ts.map