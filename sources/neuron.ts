import { Parser } from "./utils/parser.ts";
import { Random } from "./utils/random.ts";

export class Neuron {
    private bias: number;
    private weights: number[];

    constructor(
        private readonly cardinality: number,
        private readonly convergence: number,
        private readonly discrepancy: number,
    ) {
        this.bias = Random.uniform() * this.discrepancy;
        this.weights = new Array(this.cardinality);

        for (let i = 0; i < this.cardinality; i++) {
            this.weights[i] = Random.uniform() * this.discrepancy;
        }
    }

    public predict(inputs: readonly number[]): number {
        let sum = this.bias;

        for (let i = 0; i < this.cardinality; i++) {
            sum += inputs[i] * this.weights[i];
        }
        
        return Parser.sigm(sum);
    }

    public train(inputs: readonly number[], expect: number): void {
        const error = expect - this.predict(inputs);
        const delta = error * this.convergence;

        this.bias += delta;

        for (let i = 0; i < this.cardinality; i++) {
            this.weights[i] += delta * inputs[i];
        }
    }
}