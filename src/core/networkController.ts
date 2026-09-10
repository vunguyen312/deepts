import { NeuralNetwork, FrozenNetwork } from "./neuralNetwork";
import { FrozenLayer, FrozenLinear, Layer, loadLinear } from "./layers";
import { loadActivation } from "../math/activations";
import type { FrozenActivation } from "../math/activations";
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

const loadLayer = (layer: FrozenLayer): Layer => {
    const layerMap = {
        "linear": (frozen: FrozenLayer) => 
            loadLinear(frozen as FrozenLinear),
        "activation": (frozen: FrozenLayer) => 
            loadActivation(frozen as FrozenActivation)
    };

    return layerMap[layer.layerType](layer);
}

export const loadNetwork = (frozenNetwork: FrozenNetwork): NeuralNetwork => {
    const layers: Layer[] = [];
    for (const layer of frozenNetwork.layers) {
        const newLayer = loadLayer(layer);
        layers.push(newLayer);
    }

    return new NeuralNetwork(layers);
}
