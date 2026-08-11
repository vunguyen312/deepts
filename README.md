# deep.ts

<p align="center">
    <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=TypeScript&logoColor=white" />
    <img src="https://img.shields.io/badge/Node.js-18%2B-green?logo=node.js&logoColor=white" />
    <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <b> A tiny deep learning library written in TypeScript.</b>
</p>

## About

This is a small learning project built for understanding the fundamentals of deep learning.
It provides a top-level overview of neural network construction.

I know what you're thinking: What kind of nutjob would write a deep learning library from 
scratch in TypeScript? Yes, I'm that sicko. This was quite a fun project so I regret nothing!

This project is ideal for:

-   People new to deep learning looking to learn the basics
-   TypeScript developers looking for a high level deep learning experience

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vunguyen312/deepts.git
cd deepts
```

### 2. Install Dependencies
```bash
npm install
```

## Examples

### XOR Neural Network
Below is an example of a small 3-layer neural network trained to solve the XOR problem
```typescript
import NetworkController from "./NetworkController";
import { Layer } from "./neuralNetwork";

const network = NetworkController.createNetwork(
    [
        new Layer("sigmoid", 2, 3), 
        new Layer("sigmoid", 3, 1)
    ], 
    0.1
);

for (let i = 0; i < 20000; i++) {
    network.train([0, 1], [1]);
    network.train([1, 1], [0]);
    network.train([1, 0], [1]);
    network.train([0, 0], [0]);
}

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
NetworkController.freezeToJSON(network, "./src/models/weights.json");
```

## Contributing

Contributions are welcome.

1.  Fork the repository
2.  Create a new branch
3.  Commit your changes
4.  Push to your branch
5.  Open a pull request

## License

This project is licensed under the MIT License.
