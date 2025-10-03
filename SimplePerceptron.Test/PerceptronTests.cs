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
        var random = new Random(456);
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
        var random = new Random(789);
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
}
