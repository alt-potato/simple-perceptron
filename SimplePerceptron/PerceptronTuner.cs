using System.Text.Json;
using static SimplePerceptron.ProblemDefinitions;

namespace SimplePerceptron;

public class PerceptronTuner(
    PerceptronTuner.SearchSpace searchSpace,
    List<(double[] inputs, double[] targets)> trainingData,
    List<(double[] inputs, double[] targets)> testingData,
    Random? random = null
)
{
    /// <summary>
    /// Represents the search space for the genetic algorithm.
    /// </summary>
    /// <param name="Structures">A list of possible network structures.</param>
    /// <param name="LearningRateRange">The range of learning rates to search over.</param>
    /// <param name="EpochsRange">The range of epochs to search over.</param>
    /// <param name="ActivationCombos">The possible combinations of activation functions.</param>
    public record SearchSpace(
        List<int[]> Structures,
        (double Min, double Max) LearningRateRange,
        (int Min, int Max) EpochsRange,
        List<ActivationFunctions.FunctionType[]> ActivationCombos
    );

    private readonly SearchSpace _searchSpace = searchSpace;
    private readonly List<(double[] inputs, double[] targets)> _trainingData = trainingData;
    private readonly List<(double[] inputs, double[] targets)> _testingData = testingData;
    private readonly Random _random = random ?? new Random();

    public List<PerceptronProblemConfig> Tune(
        int populationSize = 20,
        double mutationRate = 0.1,
        int elitism = 2,
        CancellationToken cancellationToken = default
    )
    {
        var population = InitializePopulation(populationSize);
        List<PerceptronProblemConfig> bestPopulationSoFar = [];

        try
        {
            for (int i = 1; ; i++) // Loop indefinitely until cancelled
            {
                cancellationToken.ThrowIfCancellationRequested();

                Console.WriteLine($"\nGeneration {i}");
                EvaluatePopulation(population, cancellationToken);

                population = [.. population.OrderBy(p => p.Score)];

                // If the new best is better than any previous best, update our results
                if (
                    !bestPopulationSoFar.Any()
                    || population.First().Score < bestPopulationSoFar.First().Score
                )
                {
                    bestPopulationSoFar = [.. population];
                }

                Console.WriteLine(
                    $"Best score in generation {i}: {population.First().Score:F6} (Overall best: {bestPopulationSoFar.First().Score:F6})"
                );

                var newPopulation = new List<PerceptronProblemConfig>();
                newPopulation.AddRange(population.Take(elitism));

                while (newPopulation.Count < populationSize)
                {
                    var parent1 = population[_random.Next(population.Count / 2)];
                    var parent2 = population[_random.Next(population.Count / 2)];
                    var child = Crossover(parent1, parent2);

                    if (_random.NextDouble() < mutationRate)
                    {
                        child = Mutate(child);
                    }
                    newPopulation.Add(child);
                }
                population = newPopulation;
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n\n--- Tuning cancelled. Returning best result found so far. ---");
        }

        return bestPopulationSoFar;
    }

    private List<PerceptronProblemConfig> InitializePopulation(int size)
    {
        var population = new List<PerceptronProblemConfig>();
        for (int i = 0; i < size; i++)
        {
            population.Add(
                new PerceptronProblemConfig(
                    Structure: _searchSpace.Structures[_random.Next(_searchSpace.Structures.Count)],
                    Activations: _searchSpace.ActivationCombos[
                        _random.Next(_searchSpace.ActivationCombos.Count)
                    ],
                    TrainingData: _trainingData,
                    TestingData: _testingData,
                    LearningRate: _random.NextDouble()
                        * (_searchSpace.LearningRateRange.Max - _searchSpace.LearningRateRange.Min)
                        + _searchSpace.LearningRateRange.Min,
                    Epochs: _random.Next(_searchSpace.EpochsRange.Min, _searchSpace.EpochsRange.Max)
                )
            );
        }
        return population;
    }

    private void EvaluatePopulation(
        List<PerceptronProblemConfig> population,
        CancellationToken cancellationToken
    )
    {
        ParallelOptions parallelOptions = new() { CancellationToken = cancellationToken };
        try
        {
            Parallel.ForEach(
                population,
                parallelOptions,
                config =>
                {
                    if (config.Score != double.MaxValue)
                        return; // Already evaluated

                    // Validate that the number of activation functions is compatible with the structure.
                    // The number of layers is structure.Length - 1.
                    int numLayers = config.Structure.Length - 1;
                    int numActivations = config.Activations.Length;

                    // This logic mirrors the validation in the Perceptron constructor.
                    // If the configuration is invalid, assign it the worst possible score
                    // so it's not selected for the next generation.
                    if (numActivations > 2 && numActivations != numLayers)
                    {
                        config.Score = double.PositiveInfinity;
                        return;
                    }

                    var perceptron = new Perceptron(config.Structure, _random, config.Activations);
                    perceptron.Train(
                        config.TrainingData,
                        config.LearningRate,
                        config.Epochs,
                        cancellationToken: cancellationToken
                    );

                    double error = 0;
                    foreach (var (inputs, targets) in _testingData)
                    {
                        var prediction = perceptron.Predict(inputs);
                        for (int i = 0; i < targets.Length; i++)
                        {
                            error += Math.Pow(targets[i] - prediction[i], 2);
                        }
                    }
                    config.Score = error / _testingData.Count; // Mean Squared Error
                    config.ExportedState = perceptron.Export();
                }
            );
        }
        catch (OperationCanceledException)
        {
            // handled by caller
        }
    }

    private PerceptronProblemConfig Crossover(
        PerceptronProblemConfig parent1,
        PerceptronProblemConfig parent2
    )
    {
        return new PerceptronProblemConfig(
            Structure: _random.NextDouble() > 0.5 ? parent1.Structure : parent2.Structure,
            Activations: _random.NextDouble() > 0.5 ? parent1.Activations : parent2.Activations,
            TrainingData: _trainingData,
            TestingData: _testingData,
            LearningRate: _random.NextDouble() > 0.5 ? parent1.LearningRate : parent2.LearningRate,
            Epochs: _random.NextDouble() > 0.5 ? parent1.Epochs : parent2.Epochs
        );
    }

    private PerceptronProblemConfig Mutate(PerceptronProblemConfig config)
    {
        // Mutate one parameter at random
        return _random.Next(4) switch
        {
            0 => config with
            {
                Structure = _searchSpace.Structures[_random.Next(_searchSpace.Structures.Count)],
            },
            1 => config with
            {
                LearningRate =
                    _random.NextDouble()
                        * (_searchSpace.LearningRateRange.Max - _searchSpace.LearningRateRange.Min)
                    + _searchSpace.LearningRateRange.Min,
            },
            2 => config with
            {
                Epochs = _random.Next(_searchSpace.EpochsRange.Min, _searchSpace.EpochsRange.Max),
            },
            _ => config with
            {
                Activations = _searchSpace.ActivationCombos[
                    _random.Next(_searchSpace.ActivationCombos.Count)
                ],
            },
        };
    }

    private static readonly JsonSerializerOptions _jsonSerializerOptions = new()
    {
        WriteIndented = true,
    };

    public static void PrintBestConfig(PerceptronProblemConfig bestConfig)
    {
        Console.WriteLine("\n--- Best Configuration Found ---");
        Console.WriteLine($"Structure: [{string.Join(", ", bestConfig.Structure)}]");
        Console.WriteLine($"Activations: [{string.Join(", ", bestConfig.Activations)}]");
        Console.WriteLine($"Learning Rate: {bestConfig.LearningRate:F6}");
        Console.WriteLine($"Epochs: {bestConfig.Epochs}");
        Console.WriteLine(
            $"Best Score (MSE): {(bestConfig.Score == double.MaxValue ? "N/A" : bestConfig.Score):F6}"
        );

        if (bestConfig.ExportedState != null)
        {
            var jsonState = JsonSerializer.Serialize(
                bestConfig.ExportedState,
                _jsonSerializerOptions
            );
            Console.WriteLine("\n--- Exported Model State ---");
            Console.WriteLine(jsonState);
            Console.WriteLine("----------------------------");
        }
        else
        {
            Console.WriteLine("\n--- No Exported Model State Available ---");
        }
    }

    public static List<PerceptronProblemConfig> RunTuningSession(
        PerceptronProblemConfig config,
        SearchSpace searchSpace,
        int populationSize,
        Random? random,
        CancellationToken cancellationToken
    )
    {
        List<(double[] inputs, double[] targets)> trainingData = config.TrainingData;
        List<(double[] inputs, double[] targets)> testingData =
            config.TestingData ?? config.TrainingData;

        var tuner = new PerceptronTuner(searchSpace, trainingData, testingData, random);
        return tuner.Tune(
            populationSize: populationSize,
            mutationRate: 0.1,
            elitism: 2,
            cancellationToken: cancellationToken
        );
    }
}
