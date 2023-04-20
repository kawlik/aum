import { TestLamina } from "./sources/tests/lamina.test.ts";
import { TestNeuron } from "./sources/tests/neuron.test.ts";


/*  Defaults
/*   *   *   *   *   *   *   *   *   *   */

const USE_LAMINA    =   !false;
const PRT_REPORT    =   false;

const CONVERGANCE   =   0.1;
const DISCREPANCY   =   1.0;

const NUM_BATCHS    =   1_000;
const NUM_EPOCHS    =   1_000_000;


/*  Program
/*   *   *   *   *   *   *   *   *   *   */

let perceptronBenchmark: TestLamina | TestNeuron;

// Select layer
if (USE_LAMINA) {
    perceptronBenchmark = new TestLamina(4, 3, CONVERGANCE, DISCREPANCY);
} else {
    perceptronBenchmark = new TestNeuron(4, CONVERGANCE, DISCREPANCY);
}

// Before training
perceptronBenchmark.generateRaport(NUM_BATCHS, PRT_REPORT);

// Training
perceptronBenchmark.trainPerceptron(NUM_EPOCHS);

// After training
perceptronBenchmark.generateRaport(NUM_BATCHS, PRT_REPORT);
