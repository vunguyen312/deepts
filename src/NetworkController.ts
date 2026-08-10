import { BaseLayer, NeuralNetwork } from './neuralNetwork';
import { writeFile } from 'fs';

export default class NetworkController {
    private network: NeuralNetwork;

    public constructor(network: NeuralNetwork) {
        this.network = network;
    }

    public createNetwork(
        layers: BaseLayer[],
        learningRate: number,
        factory: (layers: BaseLayer[], learningRate: number) => NeuralNetwork,
    ): void {
        this.network = factory(layers, learningRate);
    }

    public freezeToJSON(): void {
        const frozenNetwork = this.network.freeze();
        const jsonNetwork = JSON.stringify(frozenNetwork, null, 4);

        writeFile('./src/weights/weights.json', jsonNetwork, 'utf8', err => {
            if (err) {
                console.error('There was a problem saving weights to JSON!');
                return;
            }
            console.log('Weights successfully saved.');
        });
    }
}
