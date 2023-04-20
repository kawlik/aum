import { iris2dataset, iris2spicies } from "./sources/data/iris2.ts";
import { iris3dataset, iris3spicies } from "./sources/data/iris3.ts";

import { Lamina } from "./sources/lamina.ts";
import { Random } from "./sources/utils/random.ts";

const NUM_INPUTS = 4;
const NUM_OUTPUTS = 3;
const CONVERGENCE = 0.1;
const DISCREPENCY = 5;

const dataset = iris3dataset;
const spicies = iris3spicies;

const perceptron = new Lamina(NUM_INPUTS, NUM_OUTPUTS, CONVERGENCE, DISCREPENCY);

log(25);

for (let i = 0; i < 100_000_000; i++) {
    const idx = Random.range(0, dataset.length - 1);
    const exp = new Array(NUM_OUTPUTS).fill(0);

    exp[spicies.indexOf(dataset[idx].o)] = 1;

    perceptron.train(dataset[idx].i, exp)
}

log(25);

function log(batch: number, debug = false): void {
    let correct = 0;

    console.count('\n*** REPORT *** :');
    for (let i = 0; i < batch; i++) {
        const idx = Random.range(0, dataset.length - 1);
        const exp = new Array(NUM_OUTPUTS).fill(0);
    
        exp[spicies.indexOf(dataset[idx].o)] = 1;

        const res = perceptron.predict(dataset[idx].i).map(v => v > .8 ? 1 : 0);

        if (debug) console.log(idx, '\t', exp, res);

        if (JSON.stringify(res) === JSON.stringify(exp)) {
            correct++;
        }
    }

    console.log('Accuracy: ', (100 * correct / batch).toFixed(2) + '%');
}