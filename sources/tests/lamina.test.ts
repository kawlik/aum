import { iris3dataset, iris3spicies } from "../data/iris3.ts";

import { Lamina } from "../lamina.ts";
import { Random } from "../utils/random.ts";

export class TestLamina {
    private perceptron: Lamina;

    private readonly dataset = iris3dataset;
    private readonly spicies = iris3spicies;

    constructor(
        numInputs: number,
        numOutputs: number,
        convergence: number,
        discrepancy: number,
    ) {
        this.perceptron = new Lamina(numInputs, numOutputs, convergence, discrepancy);
    }

    public trainPerceptron(epoch: number): void {
        for (let i = 0; i < epoch; i++) {
            const idx = Random.range(0, this.dataset.length - 1);

            const inputs = this.dataset[idx].inputs;
            const expect = this.getExpectedOutput(this.spicies.indexOf(this.dataset[idx].output));

            this.perceptron.train(inputs, expect);
        }
    }

    public generateRaport(batchSize: number, fullLog = false): void {
        let correctGuess = 0;

        console.count('  *** REPORT *** :');

        for (let i = 0; i < batchSize; i++) {
            const idx = Random.range(0, this.dataset.length - 1);

            const { inputs, output } = this.dataset[idx];

            const expect = this.getExpectedOutput(this.spicies.indexOf(output))
            const result = this.perceptron.predict(inputs);

            if (fullLog) this.printFullLog(idx, expect, result);

            if (this.isValidPredict(expect, result)) {
                correctGuess++;
            }
        }

        console.log('Batch size:\t', batchSize);
        console.log('Valid guess:\t', correctGuess);
        console.log('Ratio (%):\t', (100 * correctGuess / batchSize).toFixed(2) + '%');
    }

    private getExpectedOutput(index: number): number[] {
        const output = new Array(this.spicies.length).fill(0);

        output[index] = 1;

        return output;
    }

    private getPredictLabel(result: number[]): string {
        const predict = result.map(Math.round).reduce((p, q) => p + q);

        if (predict === 1) {
            const maxIndex = result.map(Math.round).indexOf(1);

            console.log(maxIndex);

            return this.spicies[maxIndex];
        }

        return 'unknown';
    }


    private isValidPredict(expect: number[], result: number[]): boolean {
        return JSON.stringify(expect) === JSON.stringify(result.map(Math.round));
    }

    private printFullLog(index: number, expect: number[], result: number[]): void {
        const { inputs, output } = this.dataset[index];
        
        const predict = this.getPredictLabel(result);

        console.log();
        console.log('inputs:', inputs,)
        console.log('\t', ' expected output:', output, '\t expected output (numeric):', expect);
        console.log('\t', 'predicted output:', predict, '\tpredicted output (numeric):', result);
        console.log();
    }
}