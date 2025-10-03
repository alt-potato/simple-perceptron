namespace SimplePerceptron;

public class Program
{
    public static void Main(string[] args)
    {
        Random random = new();

        if (args.Length < 1)
        {
            Console.WriteLine("Please provide a problem type.");
            return;
        }

        switch (args[0].ToLower())
        {
            case "xor":
                RunXorProblem(
                    new PerceptronArgs([2, 2, 1], random, ActivationFunctions.FunctionType.Sigmoid),
                    null,
                    0.1,
                    10000
                );
                break;
            default:
                Console.WriteLine("Invalid problem type.");
                break;
        }
    }

    public record PerceptronArgs(
        int[]? Structure,
        Random? Random,
        params ActivationFunctions.FunctionType[] Activations
    );

    public static void RunXorProblem(
        PerceptronArgs? args = null,
        List<(double[] inputs, double[] targets)>? data = null,
        double learningRate = 0.1,
        int epochs = 10000
    )
    {
        // setup default values
        args ??= new PerceptronArgs([2, 2, 1], null);
        int[] structure = args.Structure ?? [2, 2, 1];
        ActivationFunctions.FunctionType[] activations = args.Activations;

        Perceptron perceptron = new(structure, args.Random, activations);

        data ??= [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])];

        // training
        perceptron.Train(data, learningRate, epochs);

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
