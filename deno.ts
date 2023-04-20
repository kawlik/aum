import { TestNeuron } from "./sources/tests/neuron.test.ts";

const tn = new TestNeuron(4, 0.1, 1);

tn.generateRaport(1000);
tn.trainPerceptron(100_000);
tn.generateRaport(1000);