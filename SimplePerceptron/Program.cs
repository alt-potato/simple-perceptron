namespace SimplePerceptron;

public class Program
{
    public static void Main(string[] args)
    {
        Random random = new();
        int[] structure = [2, 2, 1];
        ActivationFunctions.FunctionType[] activations = [ActivationFunctions.FunctionType.Sigmoid];

        Perceptron perceptron = new(structure, random, activations);

        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ];

        // training arc
        perceptron.Train(data, 0.1, 10000);

        // testing
        foreach (var (inputs, targets) in data)
        {
            double[] rawResult = perceptron.Predict(inputs);
            int[] finalResult = perceptron.Predict(inputs, x => x > 0.5 ? 1 : 0);

            Console.WriteLine(
                $"{inputs[0]} XOR {inputs[1]} -> {rawResult[0]:F5} (Predicted: {finalResult[0]}, Target: {targets[0]})"
            );
        }
    }
}
