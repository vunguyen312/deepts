export * from "./core/networkController";
export { Tensor } from "./math/Tensor";
export { Layer, NeuralNetwork } from "./core/neuralNetwork";
export { SGD } from "./core/optimizer";
export { MNISTParser } from "./utils/MNISTParser";

import "./examples/mnist";
import "./examples/xor";