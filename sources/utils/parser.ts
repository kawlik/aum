export abstract class Parser {
    public static relu(value: number): number {
        return Math.max(value, 0);
    }

    public static sigm(value: number): number {
        return 1 / (1 + Math.exp(-value));
    }
}
