<p align="center">
  <image src="assets/deepts.webp" width=256 alt="deep.ts Logo">
<p>

# deep.ts

<p align="center">
    <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=TypeScript&logoColor=white" />
    <img src="https://img.shields.io/badge/Node.js-18%2B-green?logo=node.js&logoColor=white" />
    <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <b> A tiny deep learning framework written in TypeScript.</b>
</p>

------------------------------------------------------------------------

## Table of Contents

-   [About](#about)
-   [Installation](#installation)
-   [Examples](#examples)
-   [Contributing](#contributing)
-   [License](#license)

------------------------------------------------------------------------

## About

This is a small learning project built for understanding the fundamentals of deep learning.
It provides a top-level overview of neural network construction.

I know what you're thinking: What kind of nutjob would write a deep learning framework from 
scratch in TypeScript? Yes, I'm that sicko. This was quite a fun project so I regret nothing!

This project is ideal for:

-   People new to deep learning looking to learn the basics
-   TypeScript developers looking for a high level deep learning experience

------------------------------------------------------------------------

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

### 3. Download the MNIST Dataset (Optional)
The MNIST data is not committed to the repo (52 MB of binaries). Fetch it with:
```bash
npm run data
# or: ./scripts/download-mnist.sh
```
Only needed for the MNIST example; the XOR example works without it.

## Examples

### XOR Neural Network
Below is an example of a small 3-layer neural network trained to solve the XOR problem
```typescript
import { join } from "node:path";
import NetworkController from "../core/NetworkController";
import { Layer } from "../core/neuralNetwork";
import { SGDOptimizer } from "../core/optimizer";

const NUM_EPOCHS = 20000;

const network = NetworkController.createNetwork(
    [
        new Layer("sigmoid", 2, 3), 
        new Layer("sigmoid", 3, 1)
    ]
);
const optimizer = new SGDOptimizer(network.getNeurons(), 0.4);

const in1 = new Float32Array([1, 0]);
const in2 = new Float32Array([0, 0]);
const in3 = new Float32Array([1, 1]);
const in4 = new Float32Array([0, 1]);
const ex1 = new Float32Array([1]);
const ex2 = new Float32Array([0]);

for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    optimizer.zeroGrad();
    network.backward(in1, ex1);
    network.backward(in2, ex2);
    network.backward(in3, ex2);
    network.backward(in4, ex1);
    optimizer.step();
}

NetworkController.freezeToJSON(network, join(__dirname, "../weights/xor.json"));
```

### MNIST Neural Network
Below is an example of a network trained on the MNIST dataset
```typescript
import { join } from "node:path";
import NetworkController from "../core/NetworkController";
import { Layer } from "../core/neuralNetwork";
import { SGDOptimizer } from "../core/optimizer";
import MNISTParser from "../utils/MNISTParser";

const BATCH_SIZE = 64;
const NUM_EPOCHS = 30;

const network = NetworkController.createNetwork(
    [
        new Layer("relu", 784, 30),
        new Layer("sigmoid", 30, 10)
    ]
);
const optimizer = new SGDOptimizer(network.getNeurons(), 0.00625);

const trainingSet = new MNISTParser(
    join(__dirname, "../data/train-images.idx3-ubyte"),
    join(__dirname, "../data/train-labels.idx1-ubyte")
);
const trainingImages = trainingSet.getImages();
const trainingLabels = trainingSet.getLabels();

for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    for (let start = 0; start < trainingImages.count; start += BATCH_SIZE) {
        const end = Math.min(start + BATCH_SIZE, trainingImages.count);

        optimizer.zeroGrad();
        for (let j = start; j < end; j++) {
            network.backward(
                trainingSet.imageAt(j),
                trainingSet.oneHot(trainingLabels[j])
            );
        }
        optimizer.step();
    }
}

NetworkController.freezeToJSON(network, join(__dirname, "../weights/mnist.json"));
```

### Loading Networks
Below is an example of a network being loaded from a frozen model JSON file
```typescript
import { readFileSync } from "fs";
import { join } from "node:path";
import NetworkController from "../core/NetworkController";

const modelPath = join(__dirname, "../weights/xor.json");
const modelJSON = readFileSync(modelPath, "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log(network.forward([1, 0]));
console.log(network.forward([0, 0]));
console.log(network.forward([1, 1]));
```

------------------------------------------------------------------------

## Contributing

Contributions are welcome.

1.  Fork the repository
2.  Create a new branch
3.  Commit your changes
4.  Push to your branch
5.  Open a pull request

------------------------------------------------------------------------

## License

This project is licensed under the MIT License.
