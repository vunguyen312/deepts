import { BaseNeuron, BaseLayer } from "../neuralNetwork";
declare class SigmoidNeuron extends BaseNeuron {
    protected activation(x: number): number;
    protected activationDerivative(x: number): number;
}
export default class SigmoidLayer extends BaseLayer {
    protected getActivationFunction(): string;
    protected spawnNeurons(): SigmoidNeuron[];
}
export {};
//# sourceMappingURL=sigmoid.d.ts.map