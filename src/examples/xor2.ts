import { Tensor } from "../math/Tensor";
import { createNetwork, freezeToJSON } from "../core/networkController";
import { LinearLayer } from "../core/layers";
import { SGD } from "../core/optimizer";
import { Sigmoid, Softplus } from "../math/activations";

const NUM_EPOCHS = 2000;
const LEARNING_RATE = 0.4;
const MOMENTUM = 0.9;

const network = createNetwork(
    [
        new LinearLayer(2, 3), 
        new Softplus(),
        new LinearLayer(3, 1),
        new Sigmoid()
    ]
);
const optimizer = new SGD(network.params, LEARNING_RATE, MOMENTUM);

const input = new Tensor([[1, 0], [0, 0], [1, 1], [0, 1]]);
const expected = new Tensor([[1], [0], [0], [1]]);

for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    optimizer.zeroGrad();
    network.backward(input, expected);
    optimizer.step();
}

const in1 = new Tensor([1, 0]);
const in2 = new Tensor([0, 0]);
const in3 = new Tensor([1, 1]);
const in4 = new Tensor([0, 1]);

console.log("-----------------------------------------");
console.log("XOR Neural Network");
console.log("Result of [1, 0]: " + network.forward(in1).data);
console.log("Result of [0, 0]: " + network.forward(in2).data);
console.log("Result of [1, 1]: " + network.forward(in3).data);
console.log("Result of [0, 1]: " + network.forward(in4).data);
console.log("-----------------------------------------");

freezeToJSON(network, "src/weights/xor2.json");