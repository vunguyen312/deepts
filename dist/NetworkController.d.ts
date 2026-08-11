import { Layer, NeuralNetwork, FrozenNetwork } from "./neuralNetwork";
export default class NetworkController {
    static createNetwork(layers: Layer[], learningRate: number): NeuralNetwork;
    static freezeToJSON(network: NeuralNetwork, path: string): Promise<void>;
    private static loadNeurons;
    static loadNetwork(frozenNetwork: FrozenNetwork): NeuralNetwork;
}
//# sourceMappingURL=NetworkController.d.ts.map