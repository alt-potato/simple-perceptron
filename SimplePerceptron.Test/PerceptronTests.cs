using System.Text.Json;

namespace SimplePerceptron.Test;

public class PerceptronTests
{
    [Fact]
    public void Constructor_ShouldCreateCorrectStructure()
    {
        // Arrange
        int[] structure = [2, 3, 1]; // 2 inputs, 1 hidden layer with 3 neurons, 1 output neuron

        // Act
        var perceptron = new Perceptron(structure);

        // Assert
        Assert.Equal(2, perceptron.Layers.Count); // Hidden layer + Output layer

        // Hidden Layer
        Assert.Equal(3, perceptron.Layers[0].Neurons.Count);
        Assert.All(perceptron.Layers[0].Neurons, neuron => Assert.Equal(2, neuron.Weights.Length)); // 2 inputs

        // Output Layer
        Assert.Single(perceptron.Layers[1].Neurons);
        Assert.All(perceptron.Layers[1].Neurons, neuron => Assert.Equal(3, neuron.Weights.Length)); // 3 inputs from hidden layer
    }

    [Fact]
    public void Train_ShouldLearnOrProblem()
    {
        // Arrange
        var random = new Random(67);
        int[] structure = [2, 1]; // 2 inputs, 1 output neuron, no hidden layer
        var perceptron = new Perceptron(
            structure,
            random,
            ActivationFunctions.FunctionType.Sigmoid
        );
        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [1]),
        ];

        // Act
        perceptron.Train(data, 0.1, 10000);

        // Assert
        Assert.True(perceptron.Predict([0, 0])[0] < 0.5);
        Assert.True(perceptron.Predict([0, 1])[0] > 0.5);
        Assert.True(perceptron.Predict([1, 0])[0] > 0.5);
        Assert.True(perceptron.Predict([1, 1])[0] > 0.5);
    }

    [Fact]
    public void Train_ShouldLearnXorProblem()
    {
        // Arrange
        var random = new Random(67);
        int[] structure = [2, 2, 1]; // 2 inputs, 1 hidden layer with 2 neurons, 1 output neuron
        var perceptron = new Perceptron(
            structure,
            random,
            ActivationFunctions.FunctionType.Sigmoid
        );
        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ];

        // Act
        perceptron.Train(data, 0.1, 10000);

        // Assert
        Assert.True(perceptron.Predict([0, 0])[0] < 0.5);
        Assert.True(perceptron.Predict([0, 1])[0] > 0.5);
        Assert.True(perceptron.Predict([1, 0])[0] > 0.5);
        Assert.True(perceptron.Predict([1, 1])[0] < 0.5);
    }

    [Fact]
    public void Reset_ShouldReinitializeWeightsAndBias()
    {
        // Arrange
        var random = new Random(67);
        int[] structure = [2, 2, 1];
        var perceptron = new Perceptron(
            structure,
            random,
            ActivationFunctions.FunctionType.Sigmoid
        );

        // Store the initial state of a neuron to compare against
        var initialWeight = perceptron.Layers[0].Neurons[0].Weights[0];
        var initialBias = perceptron.Layers[0].Neurons[0].Bias;

        // Act
        perceptron.Reset(); // This method will need to be implemented in the Perceptron class

        // Assert
        var resetWeight = perceptron.Layers[0].Neurons[0].Weights[0];
        var resetBias = perceptron.Layers[0].Neurons[0].Bias;

        // The weights and biases should be different after reset
        Assert.NotEqual(initialWeight, resetWeight);
        Assert.NotEqual(initialBias, resetBias);
    }

    [Fact]
    public void Constructor_ShouldHandleZeroActivationFunctions()
    {
        // Arrange
        int[] structure = [2, 3, 1]; // 2 inputs, 1 hidden layer with 3 neurons, 1 output neuron

        // Act
        var perceptron = new Perceptron(structure); // No activation functions provided

        // Assert
        // All layers should default to Linear activation
        Assert.All(
            perceptron.Layers[0].Neurons,
            neuron => Assert.Equal(ActivationFunctions.FunctionType.Linear, neuron.ActivationType)
        );
        Assert.All(
            perceptron.Layers[1].Neurons,
            neuron => Assert.Equal(ActivationFunctions.FunctionType.Linear, neuron.ActivationType)
        );
    }

    [Fact]
    public void Constructor_ShouldHandleSingleActivationFunction()
    {
        // Arrange
        int[] structure = [2, 3, 1];
        var activationType = ActivationFunctions.FunctionType.Sigmoid;

        // Act
        var perceptron = new Perceptron(structure, null, activationType);

        // Assert
        // All layers should use the provided single activation function
        Assert.All(
            perceptron.Layers[0].Neurons,
            neuron => Assert.Equal(activationType, neuron.ActivationType)
        );
        Assert.All(
            perceptron.Layers[1].Neurons,
            neuron => Assert.Equal(activationType, neuron.ActivationType)
        );
    }

    [Fact]
    public void Constructor_ShouldHandleTwoActivationFunctions()
    {
        // Arrange
        int[] structure = [2, 3, 1];
        var hiddenActivation = ActivationFunctions.FunctionType.ReLU;
        var outputActivation = ActivationFunctions.FunctionType.Linear;

        // Act
        var perceptron = new Perceptron(structure, null, hiddenActivation, outputActivation);

        // Assert
        // Hidden layer should use the first, output layer the second
        Assert.All(
            perceptron.Layers[0].Neurons,
            neuron => Assert.Equal(hiddenActivation, neuron.ActivationType)
        );
        Assert.All(
            perceptron.Layers[1].Neurons,
            neuron => Assert.Equal(outputActivation, neuron.ActivationType)
        );
    }

    [Fact]
    public void Constructor_ShouldHandleMultipleActivationFunctionsMatchingLayers()
    {
        // Arrange
        int[] structure = [2, 3, 2, 1]; // 2 inputs, 2 hidden layers, 1 output
        var activation1 = ActivationFunctions.FunctionType.ReLU;
        var activation2 = ActivationFunctions.FunctionType.Tanh;
        var activation3 = ActivationFunctions.FunctionType.Sigmoid;

        // Act
        var perceptron = new Perceptron(structure, null, activation1, activation2, activation3);

        // Assert
        Assert.All(
            perceptron.Layers[0].Neurons,
            neuron => Assert.Equal(activation1, neuron.ActivationType)
        );
        Assert.All(
            perceptron.Layers[1].Neurons,
            neuron => Assert.Equal(activation2, neuron.ActivationType)
        );
        Assert.All(
            perceptron.Layers[2].Neurons,
            neuron => Assert.Equal(activation3, neuron.ActivationType)
        );
    }

    [Fact]
    public void Predict_WithSelector_ShouldReturnTransformedOutput()
    {
        // Arrange
        var random = new Random(67);
        int[] structure = [2, 1];
        var perceptron = new Perceptron(
            structure,
            random,
            ActivationFunctions.FunctionType.Sigmoid
        );
        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [1]),
        ];
        perceptron.Train(data, 0.1, 10000);

        Func<double, int> selector = x => x > 0.5 ? 1 : 0;

        // Act
        int[] result00 = perceptron.Predict([0, 0], selector);
        int[] result01 = perceptron.Predict([0, 1], selector);

        // Assert
        Assert.Single(result00);
        Assert.Equal(0, result00[0]);
        Assert.Single(result01);
        Assert.Equal(1, result01[0]);
    }

    [Fact]
    public void Train_ShouldApplyGradientClipping()
    {
        // Arrange
        // Use fixed initial weights and bias for deterministic testing
        double fixedInitialWeight = 0.5; // Example fixed value
        double fixedInitialBias = 0.2; // Example fixed value

        int[] structure = [1, 1]; // Simple structure: 1 input, 1 output neuron
        // Create a neuron with manually set initial weights and bias
        var neuron = new Neuron(
            [fixedInitialWeight],
            fixedInitialBias,
            ActivationFunctions.FunctionType.Linear
        );
        // Create a perceptron and inject the neuron with fixed weights
        var perceptron = new Perceptron(structure); // Perceptron constructor will create a default neuron
        perceptron.Layers[0].Neurons[0] = neuron; // Replace with our fixed neuron

        List<(double[] inputs, double[] targets)> data = [([100.0], [50.3])];
        double learningRate = 1.0; // High learning rate to make gradients explode
        double gradientThreshold = 0.1; // Clip gradients to this magnitude

        // Initial weights/bias are now fixed
        var initialWeight = fixedInitialWeight;
        var initialBias = fixedInitialBias;

        // Act
        perceptron.Train(data, learningRate, 1, gradientThreshold); // Train for 1 epoch

        // Assert
        // The neuron variable already points to the neuron in the perceptron
        Assert.False(double.IsNaN(neuron.Weights[0]));
        Assert.False(double.IsInfinity(neuron.Weights[0]));
        Assert.False(double.IsNaN(neuron.Bias));
        Assert.False(double.IsInfinity(neuron.Bias));

        // Calculate expected updates based on clipping
        double initialOutput = (initialWeight * data[0].inputs[0]) + initialBias;
        double initialError = data[0].targets[0] - initialOutput;
        double expectedDeltaBeforeClip = initialError * neuron.GetDerivative(); // Derivative is 1 for Linear
        double expectedClippedDelta = expectedDeltaBeforeClip.ApplyClipping(gradientThreshold);

        double expectedWeightChange = learningRate * expectedClippedDelta * data[0].inputs[0];
        double expectedBiasChange = learningRate * expectedClippedDelta;

        // Assert the final values
        Assert.Equal(initialWeight + expectedWeightChange, neuron.Weights[0], 5);
        Assert.Equal(initialBias + expectedBiasChange, neuron.Bias, 5);
    }

    [Fact]
    public void Train_ShouldApplyWeightClipping()
    {
        // Arrange
        // Use fixed initial weights and bias for deterministic testing
        double fixedInitialWeight = 0.4; // Example fixed value
        double fixedInitialBias = 0.3; // Example fixed value

        int[] structure = [1, 1];
        var neuron = new Neuron(
            [fixedInitialWeight],
            fixedInitialBias,
            ActivationFunctions.FunctionType.Linear
        );
        var perceptron = new Perceptron(structure);
        perceptron.Layers[0].Neurons[0] = neuron;

        List<(double[] inputs, double[] targets)> data = [([1.0], [100.0])]; // Large target to make weights grow
        double learningRate = 100.0; // Very high learning rate
        double minWeight = -5.0;
        double maxWeight = 5.0;

        // Initial weights/bias are now fixed
        var initialWeight = fixedInitialWeight;
        var initialBias = fixedInitialBias;

        // Act
        perceptron.Train(data, learningRate, 1, null, minWeight, maxWeight); // Train for 1 epoch

        // Assert
        // The neuron variable already points to the neuron in the perceptron
        Assert.InRange(neuron.Weights[0], minWeight, maxWeight);
        Assert.InRange(neuron.Bias, minWeight, maxWeight);

        double initialOutput = (initialWeight * data[0].inputs[0]) + initialBias;
        double error = data[0].targets[0] - initialOutput;
        double delta = error * neuron.GetDerivative(); // Derivative is 1 for Linear

        double expectedWeightBeforeClip =
            initialWeight + (learningRate * delta * data[0].inputs[0]);
        double expectedBiasBeforeClip = initialBias + (learningRate * delta);

        Assert.Equal(
            expectedWeightBeforeClip.ApplyClipping(minWeight, maxWeight),
            neuron.Weights[0],
            5
        );
        Assert.Equal(expectedBiasBeforeClip.ApplyClipping(minWeight, maxWeight), neuron.Bias, 5);
    }

    [Fact]
    public void Train_WithHighLearningRateAndClipping_ShouldNotProduceNaN()
    {
        // Arrange
        var random = new Random(67);
        int[] structure = [2, 2, 1];
        var perceptron = new Perceptron(
            structure,
            random,
            ActivationFunctions.FunctionType.Sigmoid
        );
        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ];
        double highLearningRate = 100.0;
        int epochs = 100;
        double gradientThreshold = 1.0; // Enable gradient clipping
        double weightClipValue = 5.0; // Enable weight clipping

        // Act
        perceptron.Train(
            data,
            highLearningRate,
            epochs,
            gradientThreshold,
            -weightClipValue,
            weightClipValue
        );

        // Assert
        // Check all weights and biases in the network for NaN
        foreach (var layer in perceptron.Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                foreach (var weight in neuron.Weights)
                {
                    Assert.False(double.IsNaN(weight), "Weight should not be NaN.");
                }
                Assert.False(double.IsNaN(neuron.Bias), "Bias should not be NaN.");
            }
        }

        // Also check if prediction results in NaN
        var prediction = perceptron.Predict([0, 0]);
        Assert.False(double.IsNaN(prediction[0]), "Prediction should not be NaN.");
    }

    // [Fact]
    // public void Train_ShouldLearnA2PlusBProblem()
    // {
    //     // Arrange
    //     var random = new Random(123);
    //     // Using the recommended structure and activations for non-linear problems
    //     int[] structure = [2, 10, 1];
    //     var perceptron = new Perceptron(
    //         structure,
    //         random,
    //         ActivationFunctions.FunctionType.LeakyReLU, // Hidden layer activation
    //         ActivationFunctions.FunctionType.Linear // Output layer activation
    //     );

    //     // Generate a larger training dataset
    //     List<(double[] inputs, double[] targets)> trainingData =
    //     [
    //         .. Enumerable
    //             .Range(0, 10)
    //             .SelectMany(a =>
    //                 Enumerable
    //                     .Range(0, 10)
    //                     .Select(b => (new double[] { a, b }, new double[] { a * a + b }))
    //             ),
    //     ];

    //     // Act
    //     // Use a smaller learning rate and more epochs for this complex problem
    //     perceptron.Train(trainingData, 0.0002, 200000); // Increased epochs for better learning

    //     // Assert
    //     // Test a few specific cases from the problem definition
    //     Assert.InRange(perceptron.Predict([0, 0])[0], -0.5, 0.5); // Expected 0
    //     Assert.InRange(perceptron.Predict([1, 4])[0], 4.5, 5.5); // Expected 5
    //     Assert.InRange(perceptron.Predict([2, 2])[0], 5.5, 6.5); // Expected 6
    //     Assert.InRange(perceptron.Predict([0, 6])[0], 5.5, 6.5); // Expected 6
    //     Assert.InRange(perceptron.Predict([6, 2])[0], 37.5, 38.5); // Expected 38
    //     Assert.InRange(perceptron.Predict([9, 3])[0], 83.5, 84.5); // Expected 84

    //     // Test some values not in the original small test set but in the generated training data
    //     Assert.InRange(perceptron.Predict([3, 5])[0], 13.5, 14.5); // Expected 3^2 + 5 = 14
    //     Assert.InRange(perceptron.Predict([7, 8])[0], 56.5, 57.5); // Expected 7^2 + 8 = 57
    // }

    [Fact]
    public void ExportAndImport_ShouldPreserveNetworkState()
    {
        // Arrange
        int[] structure = [2, 2, 1];
        var random1 = new Random(42);
        var trainedPerceptron = new Perceptron(
            structure,
            random1,
            ActivationFunctions.FunctionType.Sigmoid
        );

        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ];
        trainedPerceptron.Train(data, 0.1, 1000);
        var predictionBeforeExport = trainedPerceptron.Predict([1, 0]);

        // Act
        // 1. Export the state of the trained network
        var exportedState = trainedPerceptron.Export();

        // 2. Create a new, untrained network with the same structure but different seed
        var random2 = new Random(99);
        var newPerceptron = new Perceptron(
            structure,
            random2,
            ActivationFunctions.FunctionType.Sigmoid
        );

        // 3. Import the state into the new network
        newPerceptron.Import(exportedState);

        // Assert
        // The predictions of the new network should now match the trained one
        var predictionAfterImport = newPerceptron.Predict([1, 0]);
        Assert.Equal(predictionBeforeExport[0], predictionAfterImport[0]);

        // Also, assert the weights and biases are identical
        for (int i = 0; i < trainedPerceptron.Layers.Count; i++)
        {
            for (int j = 0; j < trainedPerceptron.Layers[i].Neurons.Count; j++)
            {
                Assert.Equal(
                    trainedPerceptron.Layers[i].Neurons[j].Weights,
                    newPerceptron.Layers[i].Neurons[j].Weights
                );
                Assert.Equal(
                    trainedPerceptron.Layers[i].Neurons[j].Bias,
                    newPerceptron.Layers[i].Neurons[j].Bias
                );
            }
        }

        // Assert that normalization parameters were also restored
        var importedState = newPerceptron.Export();
        Assert.Equal(exportedState.InputMin, importedState.InputMin);
        Assert.Equal(exportedState.InputMax, importedState.InputMax);
        Assert.Equal(exportedState.TargetMin, importedState.TargetMin);
        Assert.Equal(exportedState.TargetMax, importedState.TargetMax);
    }

    [Fact]
    public void Import_FromJsonString_ShouldReplicateState()
    {
        // Arrange
        int[] structure = [2, 3, 1];
        var random1 = new Random(88);
        var trainedPerceptron = new Perceptron(
            structure,
            random1,
            ActivationFunctions.FunctionType.Sigmoid
        );

        List<(double[] inputs, double[] targets)> data =
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ];
        trainedPerceptron.Train(data, 0.1, 100);

        // Export the state and serialize it to JSON
        var exportedState = trainedPerceptron.Export();
        var jsonState = JsonSerializer.Serialize(exportedState);

        // Create a new, untrained perceptron with a different seed
        var random2 = new Random(101);
        var newPerceptron = new Perceptron(
            structure,
            random2,
            ActivationFunctions.FunctionType.Sigmoid
        );

        // Act
        newPerceptron.Import(jsonState);

        // Assert
        // The new perceptron's state should now match the original exported state
        var importedState = newPerceptron.Export();
        Assert.Equal(exportedState.InputMin, importedState.InputMin);
        Assert.Equal(exportedState.InputMax, importedState.InputMax);
        Assert.Equal(exportedState.TargetMin, importedState.TargetMin);
        Assert.Equal(exportedState.TargetMax, importedState.TargetMax);

        // Verify weights and biases are identical
        for (int i = 0; i < trainedPerceptron.Layers.Count; i++)
        {
            for (int j = 0; j < trainedPerceptron.Layers[i].Neurons.Count; j++)
            {
                Assert.Equal(
                    trainedPerceptron.Layers[i].Neurons[j].Weights,
                    newPerceptron.Layers[i].Neurons[j].Weights
                );
                Assert.Equal(
                    trainedPerceptron.Layers[i].Neurons[j].Bias,
                    newPerceptron.Layers[i].Neurons[j].Bias
                );
            }
        }

        // Verify functional equivalence by checking predictions
        var originalPrediction = trainedPerceptron.Predict([1, 0]);
        var newPrediction = newPerceptron.Predict([1, 0]);
        Assert.Equal(originalPrediction, newPrediction);
    }
}
