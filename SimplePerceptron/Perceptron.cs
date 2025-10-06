namespace SimplePerceptron;

public static class ActivationFunctions
{
    public enum FunctionType
    {
        Linear,
        Step,
        Signum,
        Sigmoid,
        ReLU,
        LeakyReLU,
        Tanh,
    };

    private const double ExpMax = 709; // max safe value for Math.Exp(x)

    /// <summary>
    /// A safe version of Math.Exp, to avoid inf and NaN
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private static double SafeExp(double x)
    {
        if (x > ExpMax)
            return double.MaxValue;
        if (x < -ExpMax)
            return 0;
        return Math.Exp(x);
    }

    public static Func<double, double> GetActivationFunction(
        FunctionType type,
        double alpha = 0.1
    ) =>
        type switch
        {
            FunctionType.Linear => x => x,
            FunctionType.Step => x => x > 0 ? 1 : 0,
            FunctionType.Signum => x => Math.Sign(x),
            FunctionType.Sigmoid => x =>
                x >= 0 ? 1 / (1 + SafeExp(-x)) : SafeExp(x) / (1 + SafeExp(x)),
            FunctionType.ReLU => x => Math.Max(0, x),
            FunctionType.LeakyReLU => x => Math.Max(alpha * x, x),
            FunctionType.Tanh => Math.Tanh,
            _ => throw new NotImplementedException(),
        };

    public static double GetDerivative(FunctionType type, double x, double alpha = 0.1) =>
        type switch
        {
            FunctionType.Sigmoid => x * (1 - x),
            FunctionType.ReLU => x > 0 ? 1 : 0,
            FunctionType.LeakyReLU => x > 0 ? 1 : alpha,
            FunctionType.Tanh => 1 - Math.Pow(x, 2),
            _ => 1,
        };

    public static double GetInitialValue(
        this FunctionType type,
        int inputs,
        Random? random = null,
        double defaultValue = 0.5
    ) =>
        random is null
            ? defaultValue
            : type switch
            {
                // Xavier initialization
                FunctionType.Sigmoid or FunctionType.Tanh => (random.NextDouble() * 2 - 1)
                    * Math.Sqrt(1.0 / inputs),
                // He initialization
                FunctionType.ReLU or FunctionType.LeakyReLU => (random.NextDouble() * 2 - 1)
                    * Math.Sqrt(2.0 / inputs),
                _ => random.NextDouble() * 2 - 1,
            };

    /// <summary>
    /// Gets the derivative of the activation function of the neuron at its current value.
    /// If the activation function cannot be derived, it will return 1.
    /// </summary>
    /// <param name="neuron">The neuron for which the derivative should be calculated.</param>
    /// <returns>The derivative of the activation function at the neuron's current value.</returns>
    public static double GetDerivative(this Neuron neuron) =>
        GetDerivative(neuron.ActivationType, neuron.Value);

    /// <summary>
    /// Clips a value between a minimum and maximum value.
    /// </summary>
    public static double ApplyClipping(
        this double value,
        double min = double.MinValue,
        double max = double.MaxValue
    ) => Math.Min(Math.Max(value, min), max);

    /// <summary>
    /// Clips the absolute value of a value below a magnitude. If magnitude is null, the value will not be clipped.
    /// </summary>
    public static double ApplyClipping(this double value, double? magnitude) =>
        magnitude is null ? value : value.ApplyClipping((double)-magnitude, (double)magnitude);

    /// <summary>
    /// Applies the gradient of the loss function to a neuron's delta.
    /// If a gradient threshold is provided, it will be used to clip the absolute value of the delta.
    /// </summary>
    /// <param name="neuron">The neuron to which the gradient should be applied.</param>
    /// <param name="error">The error of the loss function with respect to the neuron's output.</param>
    /// <param name="gradientThreshold">An optional value to clip the absolute value of the delta.</param>
    /// <returns>The (optionally clipped) delta value.</returns>
    public static double ApplyGradient(
        this Neuron neuron,
        double error,
        double? gradientThreshold = null
    )
    {
        double delta = error * neuron.GetDerivative();
        return delta.ApplyClipping(gradientThreshold);
    }

    /// <summary>
    /// Calculates and sets the delta for a neuron based on its distance from a target value.
    /// </summary>
    public static void CalculateSetDelta(
        this Neuron neuron,
        double target,
        double? gradientThreshold = null
    )
    {
        double error = target - neuron.Value;
        neuron.Delta = neuron.ApplyGradient(error, gradientThreshold);
    }

    /// <summary>
    /// Calculates and sets the delta for a neuron based on the next layer.
    /// </summary>
    public static void CalculateSetDelta(
        this Neuron neuron,
        Layer nextLayer,
        int neuronIndex,
        double? gradientThreshold = null
    )
    {
        // sum the deltas from the next layer, weighted by the corresponding weights
        double error = 0;
        foreach (Neuron connectedNeuron in nextLayer.Neurons)
        {
            error += connectedNeuron.Weights[neuronIndex] * connectedNeuron.Delta;
        }

        neuron.Delta = neuron.ApplyGradient(error, gradientThreshold);
    }
}

public class Neuron
{
    public double[] Weights { get; set; }
    public double Bias { get; set; }
    public double Value { get; set; }
    public double Delta { get; set; }
    public ActivationFunctions.FunctionType ActivationType { get; set; }
    public Func<double, double> ActivationFunction { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Neuron"/> class with random weights and bias.
    /// </summary>
    public Neuron(
        int inputs,
        Random? random = null,
        ActivationFunctions.FunctionType activationType = ActivationFunctions.FunctionType.Linear
    )
    {
        Weights = new double[inputs];
        for (int i = 0; i < inputs; i++)
        {
            // [-1, 1), technically
            Weights[i] = activationType.GetInitialValue(inputs, random);
        }
        Bias = activationType.GetInitialValue(inputs, random, 0);

        ActivationType = activationType;
        ActivationFunction = ActivationFunctions.GetActivationFunction(activationType);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Neuron"/> class with manually set weights and bias.
    /// </summary>
    public Neuron(
        double[] weights,
        double bias,
        ActivationFunctions.FunctionType activationType = ActivationFunctions.FunctionType.Linear
    )
    {
        Weights = weights;
        Bias = bias;
        ActivationType = activationType;
        ActivationFunction = ActivationFunctions.GetActivationFunction(activationType);
    }

    /// <summary>
    /// Calculates the output of the neuron for the given inputs.
    /// </summary>
    public double Calculate(double[] inputs)
    {
        if (inputs.Any(double.IsNaN))
            throw new Exception($"NaN detected in inputs! {string.Join(", ", inputs)}");
        if (inputs.Any(double.IsInfinity))
            throw new Exception($"Infinity detected in inputs! {string.Join(", ", inputs)}");

        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (double.IsNaN(Weights[i]) || double.IsInfinity(Weights[i]))
                throw new Exception($"Invalid weight at index {i}: {Weights[i]}");

            sum += inputs[i] * Weights[i];
        }
        sum += Bias;
        sum = sum.ApplyClipping(-1e6, 1e6); // prevent overflow in activation functions
        Value = ActivationFunction(sum);

        if (double.IsNaN(Value))
        {
            Console.WriteLine($"NaN detected! Inputs: {string.Join(", ", inputs)}");
            throw new Exception("NaN detected!");
        }

        if (double.IsInfinity(Value))
        {
            Console.WriteLine($"Infinity detected! Inputs: {string.Join(", ", inputs)}");
            throw new Exception("Infinity detected!");
        }

        return Value;
    }

    /// <summary>
    /// Resets the weights and bias of the neuron to default values (random if rng is provided, otherwise 0.5).
    /// </summary>
    public void Reset(Random? random = null)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] = random is not null ? random.NextDouble() * 2 - 1 : 0.5;
        }
        Bias = random is not null ? random.NextDouble() * 2 - 1 : 0.5;

        Value = 0;
    }
}

public class Layer
{
    public List<Neuron> Neurons { get; set; } = [];
}

/// <summary>
/// Represents a multi-layer perceptron (MLP) neural network.
///
/// To use, first create a new instance of the Perceptron class with the desired structure and activation functions.
/// <code>
/// Perceptron perceptron = new Perceptron([2, 3, 1], new Random(), ActivationFunctions.FunctionType.Sigmoid);
/// </code>
///
/// Then, use the Train method to train the network on a dataset.
/// <code>
/// perceptron.Train(data, learningRate, epochs);
/// </code>
///
/// Finally, use the Predict method to get the output of the network for a given input.
/// <code>
/// double[] outputs = perceptron.Predict(inputs);
/// </code>
///
/// Or, use the selector method to get the output of the network for a given input, where T is the type of the output.
/// <code>
/// T[] output = perceptron.Predict(inputs, selector);
/// </code>
/// </summary>
public class Perceptron
{
    public List<Layer> Layers { get; set; } = [];
    private double[]? _inputMin;
    private double[]? _inputMax;
    private double[]? _targetMin;
    private double[]? _targetMax;

    public bool Debug { get; init; } = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="Perceptron"/> class.
    /// </summary>
    /// <param name="structure">
    /// An array of integers representing the number of neurons in each layer, where the first element is the number of
    /// inputs and the last element is the number of outputs.
    /// </param>
    /// <param name="random">An optional random number generator to use for weight initialization.</param>
    /// <param name="activationFunctions">
    /// An array of activation functions to use for each layer.
    /// If only one activation function is provided, it will be used for all layers, and if two are provided,
    /// the first will be used for hidden layers and the second will be used for the output layer.
    /// </param>
    public Perceptron(
        int[] structure,
        Random? random = null,
        params ActivationFunctions.FunctionType[] activationFunctions
    )
    {
        if (activationFunctions.Length > 2 && structure.Length != activationFunctions.Length + 1)
            throw new ArgumentException(
                "The number of activation functions must match the number of layers, or be 0, 1, or 2."
            );

        // structure[0] is the number of inputs, so skip
        for (int i = 1; i < structure.Length; i++)
        {
            Layer layer = new();

            // select activation function
            ActivationFunctions.FunctionType activationType = activationFunctions.Length switch
            {
                1 => activationFunctions[0], // if only one, use given activation function
                2 => i < structure.Length - 1 ? activationFunctions[0] : activationFunctions[1], // if only two, use first for hidden layers and last for output layer
                > 2 => activationFunctions[i - 1], // otherwise, use the one corresponding to the current layer
                _ => ActivationFunctions.FunctionType.Linear, // default to linear if not defined
            };

            // create current layer
            for (int j = 0; j < structure[i]; j++)
            {
                Neuron neuron = new(structure[i - 1], random, activationType); // register number of inputs as previous layer
                layer.Neurons.Add(neuron);
            }
            Layers.Add(layer);
        }
    }

    /// <summary>
    /// Normalizes the given data to the range [0, 1].
    /// </summary>
    private static double[] Normalize(double[] data, double[] min, double[] max)
    {
        var result = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            double range = max[i] - min[i];
            // Avoid division by zero if all values in a feature are the same
            result[i] = range == 0 ? 0 : (data[i] - min[i]) / range;
        }
        return result;
    }

    /// <summary>
    /// Denormalizes the given data from the range [0, 1] to the original range.
    /// </summary>
    private static double[] Denormalize(double[] data, double[] min, double[] max)
    {
        var result = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            double range = max[i] - min[i];
            result[i] = data[i] * range + min[i];
        }
        return result;
    }

    /// <summary>
    /// Feeds the given inputs through the network and returns the outputs of the last layer.
    /// </summary>
    /// <param name="inputs">The inputs to feed through the network.</param>
    /// <returns>The outputs of the last layer.</returns>
    private double[] FeedForward(double[] inputs)
    {
        double[] current = inputs;
        foreach (Layer layer in Layers)
        {
            double[] outputs = new double[layer.Neurons.Count];

            Parallel.For(
                0,
                layer.Neurons.Count,
                i => outputs[i] = layer.Neurons[i].Calculate(current)
            );

            current = outputs;
        }
        return current;
    }

    /// <summary>
    /// Predicts the output of the network for the given inputs.
    /// </summary>
    public double[] Predict(double[] inputs)
    {
        if (_inputMin is null || _inputMax is null || _targetMin is null || _targetMax is null)
            throw new InvalidOperationException(
                "The network has not been trained yet. Please call the Train method first."
            );

        double[] normalizedInputs = Normalize(inputs, _inputMin, _inputMax);
        double[] outputs = FeedForward(normalizedInputs);

        return Denormalize(outputs, _targetMin, _targetMax);
    }

    public T[] Predict<T>(double[] inputs, Func<double, T> selector)
    {
        return [.. Predict(inputs).Select(selector)];
    }

    /// <summary>
    /// Trains the perceptron on the given inputs and targets.
    /// </summary>
    public void Train(
        List<(double[] inputs, double[] targets)> data,
        double learningRate = 0.1,
        int epochs = 1000,
        double? gradientThreshold = null, // max absolute value of the gradient
        double minWeightValue = double.MinValue, // min value of a neuron weight
        double maxWeightValue = double.MaxValue // max value of a neuron weight
    )
    {
        if (data == null || data.Count == 0)
            return;

        // 0. find min/max for normalization
        int numInputs = data[0].inputs.Length;
        _inputMin = new double[numInputs];
        _inputMax = new double[numInputs];
        for (int i = 0; i < numInputs; i++)
        {
            _inputMin[i] = data.Min(d => d.inputs[i]);
            _inputMax[i] = data.Max(d => d.inputs[i]);
        }

        int numTargets = data[0].targets.Length;
        _targetMin = new double[numTargets];
        _targetMax = new double[numTargets];
        for (int i = 0; i < numTargets; i++)
        {
            _targetMin[i] = data.Min(d => d.targets[i]);
            _targetMax[i] = data.Max(d => d.targets[i]);
        }

        List<(double[] inputs, double[] targets)> trainingData = data;
        if (data.Count > 1)
            trainingData =
            [
                .. data.Select(d =>
                    (
                        inputs: Normalize(d.inputs, _inputMin, _inputMax),
                        targets: Normalize(d.targets, _targetMin, _targetMax)
                    )
                ),
            ];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (Debug && (epoch == 0 || (epoch + 1) % (epochs / 100) == 0))
                Console.Write($"Epoch {epoch + 1}/{epochs}\r");

            foreach (var (inputs, targets) in trainingData)
            {
                // 1. feed forward to get outputs
                double[] outputs = FeedForward(inputs);

                // 2. calculate deltas for output layer
                Layer outputLayer = Layers.Last();
                for (int i = 0; i < outputLayer.Neurons.Count; i++)
                {
                    outputLayer.Neurons[i].CalculateSetDelta(targets[i], gradientThreshold);
                }

                // 3. backpropagate deltas to hidden layers (skip output layer)
                for (int i = Layers.Count - 2; i >= 0; i--)
                {
                    Layer layer = Layers[i];
                    Layer nextLayer = Layers[i + 1];

                    for (int j = 0; j < layer.Neurons.Count; j++)
                    {
                        Neuron neuron = layer.Neurons[j];

                        neuron.CalculateSetDelta(nextLayer, j, gradientThreshold);
                    }
                }

                // 4. update weights and biases
                for (int i = 0; i < Layers.Count; i++)
                {
                    Layer layer = Layers[i];
                    double[] layerInputs =
                        i == 0 ? inputs : [.. Layers[i - 1].Neurons.Select(n => n.Value)];

                    Parallel.ForEach(
                        layer.Neurons,
                        neuron =>
                        {
                            for (int j = 0; j < neuron.Weights.Length; j++)
                            {
                                neuron.Weights[j] = (
                                    neuron.Weights[j] + learningRate * neuron.Delta * layerInputs[j]
                                ).ApplyClipping(minWeightValue, maxWeightValue);
                            }
                            neuron.Bias = (neuron.Bias + learningRate * neuron.Delta).ApplyClipping(
                                minWeightValue,
                                maxWeightValue
                            );
                        }
                    );
                }
            }
        }
    }

    /// <summary>
    /// Resets the values of all neurons in the network to their default values.
    /// </summary>
    public void Reset(Random? random = null) =>
        Layers.ForEach(l => l.Neurons.ForEach(n => n.Reset(random)));

    /// <summary>
    /// Exports the current state of the perceptron, including weights, biases, and normalization parameters.
    /// </summary>
    /// <returns>A tuple containing the state of the perceptron.</returns>
    public (
        List<List<(double[] weights, double bias)>> layers,
        double[]? inputMin,
        double[]? inputMax,
        double[]? targetMin,
        double[]? targetMax
    ) Export() =>
        (
            Layers.Select(l => l.Neurons.Select(n => (n.Weights, n.Bias)).ToList()).ToList(),
            _inputMin,
            _inputMax,
            _targetMin,
            _targetMax
        );

    /// <summary>
    /// Imports a previously exported state into the perceptron.
    /// </summary>
    /// <param name="state">The perceptron state to import.</param>
    public void Import(
        (
            List<List<(double[] weights, double bias)>> layers,
            double[]? inputMin,
            double[]? inputMax,
            double[]? targetMin,
            double[]? targetMax
        ) state
    )
    {
        var (layersData, inputMin, inputMax, targetMin, targetMax) = state;

        if (layersData.Count != Layers.Count)
            throw new ArgumentException("Import data does not match the number of layers.");

        for (int i = 0; i < Layers.Count; i++)
        {
            if (layersData[i].Count != Layers[i].Neurons.Count)
                throw new ArgumentException(
                    $"Import data for layer {i} does not match the number of neurons."
                );

            for (int j = 0; j < Layers[i].Neurons.Count; j++)
            {
                var neuronData = layersData[i][j];
                var neuron = Layers[i].Neurons[j];

                if (neuronData.weights.Length != neuron.Weights.Length)
                    throw new ArgumentException(
                        $"Import data for neuron {j} in layer {i} does not match the number of weights."
                    );

                neuron.Weights = (double[])neuronData.weights.Clone();
                neuron.Bias = neuronData.bias;
            }
        }

        _inputMin = inputMin;
        _inputMax = inputMax;
        _targetMin = targetMin;
        _targetMax = targetMax;
    }
}
