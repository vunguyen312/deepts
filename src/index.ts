export { NetworkController } from "./core/NetworkController";
export { Layer, NeuralNetwork } from "./core/neuralNetwork";
export { SGDOptimizer } from "./core/optimizer";
export { MNISTParser } from "./utils/MNISTParser";

// Can delete. Just loading in examples
import "./examples/xor";
import "./examples/mnist";