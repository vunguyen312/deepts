import { Tensor } from "../math/Tensor";
import { createNetwork, freezeToJSON } from "../core/networkController";
import { Layer } from "../core/neuralNetwork";
import { SGD } from "../core/optimizer";

const NUM_EPOCHS = 20000;

const network = createNetwork(
    [
        new Layer("relu", 2, 3), 
        new Layer("sigmoid", 3, 1)
    ]
);
const optimizer = new SGD(network.params, 0.4);

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