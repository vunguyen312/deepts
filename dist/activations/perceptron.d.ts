import { BaseNeuron, BaseLayer } from "../neuralNetwork";
declare class Perceptron extends BaseNeuron {
    protected activation(x: number): number;
    protected activationDerivative(x: number): number;
}
export default class PerceptronLayer extends BaseLayer {
    protected getActivationFunction(): string;
    protected spawnNeurons(): Perceptron[];
}
export {};
//# sourceMappingURL=perceptron.d.ts.map