using static SimplePerceptron.ProblemDefinitions;

namespace SimplePerceptron;

public class Program
{
    static string OUTPUT_OVERRIDE { get; set; } = "training"; // training, testing, both
    static string OUTPUT_MODE { get; set; } = "invalid"; // all, invalid

    public static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Please provide a problem type.");
            return;
        }

        string problem = args[0].Trim().ToLower();
        if (!Problems.TryGetValue(problem, out PerceptronProblemConfig? value))
        {
            Console.WriteLine($"Unknown problem type: {problem}");
            Console.WriteLine("Available problems: " + string.Join(", ", Problems.Keys));
            return;
        }

        Random random = new();
        RunProblem(value, random);
    }

    public static void RunProblem(PerceptronProblemConfig config, Random? random = null)
    {
        Perceptron perceptron = new(config.Structure, random, config.Activations) { Debug = true };

        // training
        perceptron.Train(config.TrainingData, config.LearningRate, config.Epochs);

        // testing
        // uses training data and testing data
        List<(double[] inputs, double[] targets)> testData = OUTPUT_OVERRIDE switch
        {
            "training" => config.TrainingData,
            "testing" => config.TestingData ?? [],
            "both" => [.. config.TrainingData, .. config.TestingData ?? []],
            _ => config.TestingData ?? config.TrainingData,
        };
        foreach ((double[] inputs, double[] targets) in testData)
        {
            double[] rawResult = perceptron.Predict(inputs);
            object[]? finalResult = config.Selector is null
                ? null
                : perceptron.Predict(inputs, config.Selector);

            // build input string
            string inputString = config.InputFormat is null
                ? string.Join(" ", inputs)
                : string.Format(config.InputFormat, [.. inputs]);

            // build output string
            string outputString = "";
            if (config.OutputFormat is not null)
            {
                // use given output format
                outputString = string.Format(
                    config.OutputFormat,
                    string.Join(" ", rawResult.Select(r => r.ToString("F4"))),
                    string.Join(" ", finalResult ?? []),
                    string.Join(" ", targets)
                );
            }
            else
            {
                // use default:
                // "{raw} (Actual: {final}, Predicted: {targets})"
                // or
                // "{raw} (Predicted: {targets})"
                string rawResultString = string.Join(" ", rawResult.Select(r => r.ToString("F4")));
                string finalResultString = finalResult is null
                    ? ""
                    : $"Actual: {string.Join(" ", finalResult)}, ";
                string targetString = $"Expected: {string.Join(" ", targets)}";

                outputString = $"{rawResultString} ({finalResultString}{targetString})";
            }

            switch (OUTPUT_MODE)
            {
                case "invalid":
                    if (
                        finalResult is not null
                        && string.Join(" ", finalResult) != string.Join(" ", targets)
                    )
                        Console.WriteLine($"{inputString} -> {outputString}");
                    break;
                default:
                    Console.WriteLine($"{inputString} -> {outputString}");
                    break;
            }
        }
    }
}
