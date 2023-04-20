export abstract class Random {
    public static range(min: number, max: number): number {
        if (min > max) [min, max] = [max, min];

        return Math.floor(Math.random() * (max - min + 1));
    }

    public static uniform(): number {
        return Math.random() * 2 - 1;
    }
}