"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs_1 = require("fs");
const NetworkController_1 = __importDefault(require("./NetworkController"));
const modelJSON = (0, fs_1.readFileSync)("./src/weights/weights.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController_1.default.loadNetwork(modelData);
console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
//# sourceMappingURL=index.js.map