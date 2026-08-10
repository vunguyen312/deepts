import { BaseNeuron, BaseLayer, BaseNetwork } from './baseNetwork';
import { writeFile } from 'fs';

export default class NetworkController<T extends BaseNeuron> {
    private network: BaseNetwork<T>;

    public constructor(network: BaseNetwork<T>) {
        this.network = network;
    }

    public createNetwork(
        layers: BaseLayer<T>[],
        learningRate: number,
        factory: (layers: BaseLayer<T>[], learningRate: number) => BaseNetwork<T>,
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
