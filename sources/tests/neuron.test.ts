import { iris2dataset, iris2spicies } from "../data/iris2.ts";

import { Neuron } from "../neuron.ts";
import { Random } from "../utils/random.ts";

export class TestNeuron {
    private perceptron: Neuron;

    private readonly dataset = iris2dataset;
    private readonly spicies = iris2spicies;

    constructor(
        numInputs: number,
        convergence: number,
        discrepancy: number,
    ) {
        this.perceptron = new Neuron(numInputs, convergence, discrepancy);
    }

    public trainPerceptron(epoch: number): void {
        for (let i = 0; i < epoch; i++) {
            const { inputs, output } = this.dataset[i % this.dataset.length];

            this.perceptron.train(inputs, this.spicies.indexOf(output));
        }
    }

    public generateRaport(batchSize: number, fullLog = false): void {
        let correctGuess = 0;

        console.count('  *** REPORT *** :');

        for (let i = 0; i < batchSize; i++) {
            const idx = Random.range(0, this.dataset.length - 1);
            const res = this.perceptron.predict(this.dataset[idx].inputs);

            if (fullLog) this.printFullLog(idx, res);

            if (this.isValidPredict(idx, res)) {
                correctGuess++;
            }
        }

        console.log('Batch size:\t', batchSize);
        console.log('Valid guess:\t', correctGuess);
        console.log('Ratio (%):\t', (100 * correctGuess / batchSize).toFixed(2) + '%');
    }

    private isValidPredict(index: number, result: number): boolean {
        const guess = Math.round(result);

        const correctSpicie = this.dataset[index].output;
        const predictSpicie = this.spicies[guess];

        return correctSpicie === predictSpicie;
    }

    private printFullLog(index: number, result: number): void {
        const { inputs, output } = this.dataset[index];

        const numeric = this.spicies.indexOf(output);
        const predict = Math.round(result);

        console.log('inputs:', inputs, 'output', output, 'output (numeric)', numeric, 'predict', predict, 'predict (numeric)', result);
    }
}