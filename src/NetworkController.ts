import { Neuron, Layer, NeuralNetwork, FrozenLayer, FrozenNetwork } from "./neuralNetwork";
import { writeFile } from "fs";
import { activationMap } from "./activations";

export default class NetworkController {
    public static createNetwork(layers: Layer[], 
                                learningRate: number): NeuralNetwork {
        return new NeuralNetwork(layers, learningRate);
    }

    public static async freezeToJSON(network: NeuralNetwork, 
                                     path: string): Promise<void> {
        const frozenNetwork = network.freeze();
        const jsonNetwork = JSON.stringify(frozenNetwork, null, 4);
        
        console.log("Loading model...");
        await writeFile(path, jsonNetwork, "utf8", err => {
            if (err) {
                console.error("There was a problem saving weights to JSON!");
                return;
            }
        });
    }

    private static loadNeurons(layer: FrozenLayer): Neuron[] {
        const neurons: Neuron[] = [];
        for (const neuron of layer.neurons) {
            const newNeuron = new Neuron(activationMap[layer.activation],
                                         neuron.weights, neuron.bias);
            neurons.push(newNeuron);
        }

        return neurons;
    }

    public static loadNetwork(frozenNetwork: FrozenNetwork): NeuralNetwork {
        const layers: Layer[] = [];

        for (const layer of frozenNetwork.layers) {
            const neurons = NetworkController.loadNeurons(layer);
            const newLayer = new Layer(layer.activation, layer.inputSize, 
                                       layer.outputSize, neurons);
            layers.push(newLayer);
        }

        return new NeuralNetwork(layers, frozenNetwork.learningRate);
    }
}
