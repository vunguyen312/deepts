import { Neuron, Layer, NeuralNetwork, FrozenLayer, FrozenNetwork } from "./neuralNetwork";
import { writeFile } from "fs/promises";

export const createNetwork = (layers: Layer[]): NeuralNetwork => {
    return new NeuralNetwork(layers);
}

export const freezeToJSON = async (network: NeuralNetwork, 
                                   path: string): Promise<void> => {
    const frozenNetwork = network.freeze();
    const jsonNetwork = JSON.stringify(frozenNetwork, null, 4);
    try {
        await writeFile(path, jsonNetwork);
    } catch (error) {
        console.error("Error saving weights");
    }
}

const loadNeurons = (layer: FrozenLayer): Neuron[] => {
    const neurons: Neuron[] = [];
    for (const neuron of layer.neurons) {
        const newNeuron = new Neuron(new Float32Array(neuron.weights), 
                                     neuron.bias);
        neurons.push(newNeuron);
    }

    return neurons;
}

export const loadNetwork = (frozenNetwork: FrozenNetwork): NeuralNetwork => {
    const layers: Layer[] = [];
    for (const layer of frozenNetwork.layers) {
        const neurons = loadNeurons(layer);
        const newLayer = new Layer(layer.activation, layer.inputSize, 
                                   layer.outputSize, neurons);
        layers.push(newLayer);
    }

    return new NeuralNetwork(layers);
}
