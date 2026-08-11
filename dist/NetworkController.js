"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const neuralNetwork_1 = require("./neuralNetwork");
const fs_1 = require("fs");
const activations_1 = require("./activations");
class NetworkController {
    static createNetwork(layers, learningRate) {
        return new neuralNetwork_1.NeuralNetwork(layers, learningRate);
    }
    static async freezeToJSON(network, path) {
        const frozenNetwork = network.freeze();
        const jsonNetwork = JSON.stringify(frozenNetwork, null, 4);
        await (0, fs_1.writeFile)(path, jsonNetwork, "utf8", err => {
            if (err) {
                console.error("There was a problem saving weights to JSON!");
                return;
            }
        });
    }
    static loadNeurons(layer) {
        const neurons = [];
        for (const neuron of layer.neurons) {
            const newNeuron = new neuralNetwork_1.Neuron(activations_1.activationMap[layer.activation], neuron.weights, neuron.bias);
            neurons.push(newNeuron);
        }
        return neurons;
    }
    static loadNetwork(frozenNetwork) {
        const layers = [];
        for (const layer of frozenNetwork.layers) {
            const neurons = NetworkController.loadNeurons(layer);
            const newLayer = new neuralNetwork_1.Layer(layer.activation, layer.inputSize, layer.outputSize, neurons);
            layers.push(newLayer);
        }
        return new neuralNetwork_1.NeuralNetwork(layers, frozenNetwork.learningRate);
    }
}
exports.default = NetworkController;
//# sourceMappingURL=NetworkController.js.map