import { BaseNeuron, BaseLayer } from "../neuralNetwork";
declare class ReLUNeuron extends BaseNeuron {
    protected activation(x: number): number;
    protected activationDerivative(x: number): number;
}
export default class ReLULayer extends BaseLayer {
    protected getActivationFunction(): string;
    protected spawnNeurons(): ReLUNeuron[];
}
export {};
//# sourceMappingURL=relu.d.ts.map