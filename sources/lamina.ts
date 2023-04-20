import { Neuron } from "./neuron.ts";

export class Lamina {
    private neurons: Neuron[];

    constructor(
        private readonly numInputs: number,
        private readonly numOutputs: number,
        private readonly convergence: number,
        private readonly discrepancy: number,
    ) {
        this.neurons = new Array(this.numOutputs);

        for (let i = 0; i < this.numOutputs; i++) {
            this.neurons[i] = new Neuron(this.numInputs, convergence, discrepancy);
        }
    }

    public predict(inputs: readonly number[]): number[] {
        const vect = new Array(this.numOutputs);

        for (let i = 0; i < this.numOutputs; i++) {
            vect[i] = this.neurons[i].predict(inputs);
        }

        return vect;
    }

    public train(inputs: readonly number[], expect: readonly number[]): void {
        for (let i = 0; i < this.numOutputs; i++) {
            this.neurons[i].train(inputs, expect[i]);
        }
    }
}