using System.Security.Cryptography;
using static SimplePerceptron.ProblemDefinitions;

namespace SimplePerceptron;

public class Program
{
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

        RunProblem(value);
    }

    public static void RunProblem(PerceptronProblemConfig config, Random? random = null)
    {
        Perceptron perceptron = new(config.Structure, random, config.Activations);

        // training
        perceptron.Train(config.TrainingData, config.LearningRate, config.Epochs);

        // testing
        // uses training data and testing data
        foreach (
            (double[] inputs, double[] targets) in config.TrainingData.Concat(
                config.TestingData ?? []
            )
        )
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
                string targetString = $"Predicted: {string.Join(" ", targets)}";

                outputString = $"{rawResultString} ({finalResultString}{targetString})";
            }

            Console.WriteLine($"{inputString} -> {outputString}");
        }
    }
}
