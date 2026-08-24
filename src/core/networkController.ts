import { Layer, NeuralNetwork, FrozenNetwork } from "./neuralNetwork";
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

export const loadNetwork = (frozenNetwork: FrozenNetwork): NeuralNetwork => {
    const layers: Layer[] = [];
    for (const layer of frozenNetwork.layers) {
        const floatWeights = new Float32Array(layer.weights);
        const floatBiases = new Float32Array(layer.biases);
        const newLayer = new Layer(layer.activation, layer.inputSize, 
                                   layer.outputSize, floatWeights, 
                                   floatBiases);
        layers.push(newLayer);
    }

    return new NeuralNetwork(layers);
}
