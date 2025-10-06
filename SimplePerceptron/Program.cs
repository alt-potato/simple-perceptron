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

        // Set up a 10-minute total time limit and allow user cancellation via Ctrl+C
        using var totalTimeCts = new CancellationTokenSource(TimeSpan.FromMinutes(10));
        using var userCancelCts = new CancellationTokenSource();
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            totalTimeCts.Token,
            userCancelCts.Token
        );

        Console.CancelKeyPress += (_, eventArgs) =>
        {
            Console.WriteLine("\nCtrl+C detected. Stopping tuning process...");
            userCancelCts.Cancel();
            eventArgs.Cancel = true; // Prevent the process from terminating immediately
        };

        try
        {
            if (args.Length > 1)
            {
                switch (args[1].Trim().ToLower())
                {
                    case "tune":
                        TuneProblem(value, random: random, cancellationToken: linkedCts.Token);
                        break;
                    case "phasedtune":
                        TuneProblemPhased(
                            value,
                            random: random,
                            cancellationToken: linkedCts.Token
                        );
                        break;
                    default:
                        RunProblem(value, random);
                        break;
                }
            }
            else
            {
                RunProblem(value, random);
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\nOperation was cancelled globally due to timeout or user request.");
        }
        finally
        {
            totalTimeCts.Dispose();
            userCancelCts.Dispose();
            linkedCts.Dispose();
            Console.WriteLine("\nDone!");
        }
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

    /// <summary>
    /// Tunes a network for the given problem.
    /// </summary>
    public static void TuneProblem(
        PerceptronProblemConfig config,
        PerceptronTuner.SearchSpace? inputSearchSpace = null,
        Random? random = null,
        CancellationToken cancellationToken = default
    )
    {
        // default to generic search space
        PerceptronTuner.SearchSpace searchSpace =
            inputSearchSpace
            ?? new(
                Structures:
                [
                    [2, 2, 1],
                    [2, 3, 1],
                    [2, 4, 1],
                    [2, 4, 2, 1],
                    [2, 8, 4, 1],
                ],
                LearningRateRange: (1e-5, 0.2),
                EpochsRange: (5000, 20000),
                ActivationCombos:
                [
                    [ActivationFunctions.FunctionType.Sigmoid],
                    [
                        ActivationFunctions.FunctionType.LeakyReLU,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                    [
                        ActivationFunctions.FunctionType.ReLU,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                    [
                        ActivationFunctions.FunctionType.Tanh,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                ]
            );

        var results = PerceptronTuner.RunTuningSession(
            config,
            searchSpace,
            10, // population size
            random,
            cancellationToken
        );

        if (results.Count == 0)
        {
            Console.WriteLine("Tuning was cancelled before any results could be determined.");
            return;
        }

        var bestConfig = results.First();
        PerceptronTuner.PrintBestConfig(bestConfig);
    }

    /// <summary>
    /// Tunes a network for the given problem using a multiphase approach.
    ///
    /// Phase 1: broad search (many structures, few epochs)
    ///   Looks for promising structures that work well with few epochs
    /// Phase 2: refined search (few structures, many epochs)
    ///   Out of the promising structures from phase 1, looks for the best one
    /// </summary>
    public static void TuneProblemPhased(
        PerceptronProblemConfig config,
        PerceptronTuner.SearchSpace? inputSearchSpace = null,
        Random? random = null,
        CancellationToken cancellationToken = default
    )
    {
        var explorationTime = TimeSpan.FromMinutes(3);
        var refinementTime = TimeSpan.FromMinutes(7);
        const int POPULATION_SIZE = 20;
        const int NUM_TOP_CANDIDATES = 3;

        List<(double[] inputs, double[] targets)> trainingData = config.TrainingData;
        List<(double[] inputs, double[] targets)> testingData =
            config.TestingData ?? config.TrainingData;

        // --- phase 1: broad search ---
        Console.WriteLine(
            $"\n--- Starting Phase 1: Exploration ({explorationTime.TotalMinutes} mins) ---"
        );
        using var explorationCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken
        );
        explorationCts.CancelAfter(explorationTime);

        PerceptronTuner.SearchSpace explorationSearchSpace =
            inputSearchSpace
            ?? new(
                Structures:
                [
                    [2, 2, 1],
                    [2, 3, 1],
                    [2, 4, 1],
                    [2, 4, 2, 1],
                    [2, 8, 4, 1],
                    [2, 10, 5, 1],
                ],
                LearningRateRange: (1e-4, 0.2),
                EpochsRange: (1000, 5000), // Lower epochs for faster evaluation
                ActivationCombos:
                [
                    [ActivationFunctions.FunctionType.Sigmoid],
                    [
                        ActivationFunctions.FunctionType.LeakyReLU,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                    [
                        ActivationFunctions.FunctionType.ReLU,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                    [
                        ActivationFunctions.FunctionType.Tanh,
                        ActivationFunctions.FunctionType.Sigmoid,
                    ],
                ]
            );

        var explorationResults = PerceptronTuner.RunTuningSession(
            config,
            explorationSearchSpace,
            POPULATION_SIZE,
            random,
            explorationCts.Token
        );

        if (explorationResults.Count == 0)
        {
            Console.WriteLine("Phase 1 was cancelled or timed out before any results were found.");
            return;
        }

        var topCandidates = explorationResults.Take(NUM_TOP_CANDIDATES).ToList();
        Console.WriteLine("\n--- Phase 1 Complete ---");
        Console.WriteLine("Top candidates found:");
        foreach (var candidate in topCandidates)
        {
            Console.WriteLine(
                $"  - Score: {candidate.Score:F6}, Structure: [{string.Join(", ", candidate.Structure)}], Activations: [{string.Join(", ", candidate.Activations)}]"
            );
        }

        // if main cancellation token was triggered, stop before phase 2
        if (cancellationToken.IsCancellationRequested)
        {
            Console.WriteLine("\nMain cancellation token triggered. Halting phased tuning.");
            PerceptronTuner.PrintBestConfig(topCandidates.First());
            return;
        }

        // --- phase 2: refined search ---
        Console.WriteLine(
            $"\n--- Starting Phase 2: Refined Search ({refinementTime.TotalMinutes} mins) ---"
        );
        using var refinementCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken
        );
        refinementCts.CancelAfter(refinementTime);

        PerceptronTuner.SearchSpace refinementSearchSpace = new(
            Structures: [.. topCandidates.Select(c => c.Structure).Distinct()],
            LearningRateRange: (1e-5, 0.1), // Narrow the learning rate
            EpochsRange: (10000, 25000), // Higher epochs for deep training
            ActivationCombos:
            [
                .. topCandidates.Select(c => c.Activations).DistinctBy(a => string.Join(",", a)),
            ]
        );

        var refinementResults = PerceptronTuner.RunTuningSession(
            config,
            refinementSearchSpace,
            POPULATION_SIZE,
            random,
            refinementCts.Token
        );

        // take best result from phase 2 if it exists, otherwise take best result from phase 1
        var bestConfig = (
            refinementResults.Count != 0 ? refinementResults : explorationResults
        ).First();

        PerceptronTuner.PrintBestConfig(bestConfig);
    }
}
