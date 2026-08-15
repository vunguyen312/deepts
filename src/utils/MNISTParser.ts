import { readFileSync } from "node:fs";

interface IdxImages {
    count: number;
    rows: number;
    cols: number;
    data: Uint8Array;
}

export default class MNISTParser {
    private readonly imagesPath: string;
    private readonly labelsPath: string;
    private readonly idxImages: IdxImages;
    private readonly idxLabels: Uint8Array;

    constructor(imagesPath: string, labelsPath: string) {
        this.imagesPath = imagesPath;
        this.labelsPath = labelsPath;
        this.idxImages = this.loadIdxImages();
        this.idxLabels = this.loadIdxLabels();
    }

    private loadIdxImages(): IdxImages {
        const buffer = readFileSync(this.imagesPath);
        const magic = buffer.readUint32BE();
        const decimalMagic = magic.toString(16);
        const EXPECTED_MAGIC = 0x00000803;
        if (magic !== EXPECTED_MAGIC) {
            throw new Error(
                `Magic number mismatch, expected 2051, got ${decimalMagic}`
            );
        }

        const COUNT_OFFSET = 4;
        const ROWS_OFFSET = 8;
        const COLS_OFFSET = 12;
        const EXPECTED_HEADER_SIZE = 16;
        const count = buffer.readUInt32BE(COUNT_OFFSET);
        const rows = buffer.readUint32BE(ROWS_OFFSET);
        const cols = buffer.readUint32BE(COLS_OFFSET);
        
        const expected = EXPECTED_HEADER_SIZE + count * rows * cols;
        if (buffer.length !== expected) {
            throw new Error(
                `Buffer length mismatch, expected ${expected},
                 got ${buffer.length}`
            );
        }
        
        const data = buffer.subarray(16);
        return {
            count,
            rows,
            cols,
            data
        };
    }

    private loadIdxLabels(): Uint8Array {
        const buffer = readFileSync(this.labelsPath);
        const magic = buffer.readUint32BE();
        const decimalMagic = magic.toString(16);
        const EXPECTED_MAGIC = 0x00000801;
        if (magic !== EXPECTED_MAGIC) {
            throw new Error(
                `Magic number mismatch, expected 2049, got ${decimalMagic}`
            );
        }

        const COUNT_OFFSET = 4;
        const EXPECTED_HEADER_SIZE = 8;
        const count = buffer.readUint32BE(COUNT_OFFSET);
        const expected = EXPECTED_HEADER_SIZE + count;
        if (buffer.length !== EXPECTED_HEADER_SIZE + count) {
            throw new Error(
                `Buffer length mismatch, expected ${expected},
                 got ${buffer.length}`
            );
        }

        const labels = buffer.subarray(8);
        return labels;
    }

    public imageAt(i: number): number[] {
        const size = this.idxImages.rows * this.idxImages.cols;
        const slice = this.idxImages.data.subarray(i * size, (i + 1) * size);
        return Array.from(slice, v => v / 255);
    }

    public oneHot(label: number): number[] {
        const NUM_DIGITS = 10;
        const expectedVec: number[] = new Array(NUM_DIGITS).fill(0);
        expectedVec[label] = 1;
        return expectedVec;
    }

    public argMax(vec: number[]): number {
        let highest = 0;
        for (let i = 0; i < vec.length; i++) {
            if (vec[i] > vec[highest]) {
                highest = i;
            }
        }

        return highest;
    }

    public printDigit(i: number): void {
        for (let j = 0; j < this.idxImages.rows; j++) {
            let line = "";
            for (let k = 0; k < this.idxImages.cols; k++) {
                const index = i * 784 + j * this.idxImages.cols + k;
                line += this.idxImages.data[index] > 127 ? "#" : ".";
            }
            console.log(line);
        }
    }

    public getImages(): IdxImages {
        return this.idxImages;
    }

    public getLabels(): Uint8Array {
        return this.idxLabels;
    }
}
