namespace SimplePerceptron;

public static class ProblemDefinitions
{
    public record PerceptronProblemConfig(
        int[] Structure,
        ActivationFunctions.FunctionType[] Activations,
        List<(double[] inputs, double[] targets)> TrainingData,
        double LearningRate,
        int Epochs,
        Func<double, object>? Selector,
        List<(double[] inputs, double[] targets)>? TestingData = null,
        string? InputFormat = null,
        string? OutputFormat = null
    )
    {
        public Type OutputType { get; init; } = typeof(double);
    };

    public static readonly Dictionary<string, PerceptronProblemConfig> Problems = new()
    {
        {
            "or",
            new PerceptronProblemConfig(
                Structure: [2, 1], // Simpler structure for OR, often doesn't need a hidden layer
                Activations: [ActivationFunctions.FunctionType.Sigmoid],
                TrainingData: [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [1])],
                LearningRate: 0.1,
                Epochs: 10000,
                Selector: x => x > 0.5 ? 1 : 0,
                InputFormat: "{0} OR {1}"
            )
        },
        {
            "xor",
            new PerceptronProblemConfig(
                Structure: [2, 2, 1],
                Activations: [ActivationFunctions.FunctionType.Sigmoid],
                TrainingData: [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])],
                LearningRate: 0.1,
                Epochs: 10000,
                Selector: x => x > 0.5 ? 1 : 0,
                InputFormat: "{0} XOR {1}"
            )
        },
    };
}
