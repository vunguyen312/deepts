import { BaseLayer, NeuralNetwork } from './neuralNetwork';
export default class NetworkController {
    private network;
    constructor(network: NeuralNetwork);
    createNetwork(layers: BaseLayer[], learningRate: number, factory: (layers: BaseLayer[], learningRate: number) => NeuralNetwork): void;
    freezeToJSON(): void;
}
//# sourceMappingURL=NetworkController.d.ts.map