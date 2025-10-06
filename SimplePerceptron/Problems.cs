namespace SimplePerceptron;

public static class ProblemDefinitions
{
    public record PerceptronProblemConfig(
        int[] Structure,
        ActivationFunctions.FunctionType[] Activations,
        List<(double[] inputs, double[] targets)> TrainingData,
        double LearningRate,
        int Epochs,
        double? GradientThreshold = null,
        double MinWeightValue = double.MinValue,
        double MaxWeightValue = double.MaxValue,
        List<(double[] inputs, double[] targets)>? TestingData = null,
        Func<double, object>? Selector = null,
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
                Structure: [2, 1],
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
        {
            "a^2+b",
            new PerceptronProblemConfig(
                Structure: [2, 8, 8, 1], // [2, .. Enumerable.Repeat(100, 50), 1],
                Activations:
                [
                    ActivationFunctions.FunctionType.LeakyReLU,
                    ActivationFunctions.FunctionType.Linear,
                ],
                TrainingData:
                [
                    .. Enumerable
                        .Range(0, 20)
                        .SelectMany(a =>
                            Enumerable
                                .Range(0, 20)
                                .Select(b => (new double[] { a, b }, new double[] { a * a + b }))
                        ),
                ],
                // [([0, 0], [0])],
                TestingData:
                [
                    ([0, 0], [0]),
                    ([1, 4], [5]),
                    ([2, 2], [6]),
                    ([0, 6], [6]),
                    ([6, 2], [38]),
                    ([9, 3], [84]),
                ],
                LearningRate: 5e-4,
                Epochs: 200000,
                GradientThreshold: 1,
                MinWeightValue: -100,
                MaxWeightValue: 100,
                Selector: x => (int)(x + 0.5),
                InputFormat: "{0}^2 + {1}"
            )
        },
    };
}
