"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const fs_1 = require("fs");
class NetworkController {
    network;
    constructor(network) {
        this.network = network;
    }
    createNetwork(layers, learningRate, factory) {
        this.network = factory(layers, learningRate);
    }
    freezeToJSON() {
        const frozenNetwork = this.network.freeze();
        const jsonNetwork = JSON.stringify(frozenNetwork, null, 4);
        (0, fs_1.writeFile)('./src/weights/weights.json', jsonNetwork, 'utf8', err => {
            if (err) {
                console.error('There was a problem saving weights to JSON!');
                return;
            }
            console.log('Weights successfully saved.');
        });
    }
}
exports.default = NetworkController;
//# sourceMappingURL=NetworkController.js.map