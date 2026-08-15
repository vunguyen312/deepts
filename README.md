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

### 3. Download the MNIST Dataset
The MNIST data is not committed to the repo (52 MB of binaries). Fetch it with:
```bash
npm run data
# or: ./scripts/download-mnist.sh
```
Only needed for the MNIST example; the XOR example works without it.

## Examples

Runnable versions of these examples live in `src/examples/` — run them with
`npx tsx src/examples/xor.ts` or `npm run dev` (runs both).

### XOR Neural Network
Below is an example of a small 3-layer neural network trained to solve the XOR problem
```typescript
import NetworkController from "../core/NetworkController";
import { Layer } from "../core/neuralNetwork";

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
NetworkController.freezeToJSON(network, "./src/weights/xor.json");
```

### MNIST Neural Network
Below is an example of a network trained on the MNIST dataset
```typescript
import NetworkController from "../core/NetworkController";
import { Layer } from "../core/neuralNetwork";
import MNISTParser from "../utils/MNISTParser";

const network = NetworkController.createNetwork(
    [
        new Layer("relu", 784, 30),
        new Layer("sigmoid", 30, 10)
    ],
    0.1
);

const trainingSet = new MNISTParser(
    "src/data/train-images.idx3-ubyte",
    "src/data/train-labels.idx1-ubyte"
);
const trainingImages = trainingSet.getImages();
const trainingLabels = trainingSet.getLabels();

for (let i = 0; i < 30; i++) {
    for (let j = 0; j < trainingImages.count; j++) {
        const currImage = trainingSet.imageAt(j);
        const currExpected = trainingSet.oneHot(trainingLabels[j]);
        network.train(currImage, currExpected);
    }
}

const testSet = new MNISTParser(
    "src/data/t10k-images.idx3-ubyte",
    "src/data/t10k-labels.idx1-ubyte"
);

const test = network.forwardPass(testSet.imageAt(0));
console.log("Network saw " + testSet.argMax(test));
console.log("Expected is " + testSet.getLabels()[0]);

NetworkController.freezeToJSON(network, "./src/weights/mnist.json");
```

### Loading Networks
Below is an example of a network being loaded from a frozen model JSON file
```typescript
import { readFileSync } from "fs";
import NetworkController from "../core/NetworkController";

const modelJSON = readFileSync("./src/weights/xor.json", "utf-8");
const modelData = JSON.parse(modelJSON);
const network = NetworkController.loadNetwork(modelData);

console.log(network.forwardPass([1, 0]));
console.log(network.forwardPass([0, 0]));
console.log(network.forwardPass([1, 1]));
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
