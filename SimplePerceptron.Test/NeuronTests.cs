namespace SimplePerceptron.Test;

public class NeuronTests
{
    [Theory]
    [InlineData(new double[] { 0.5, -0.5 }, 0.2, new double[] { 1, 1 }, 0.2)] // 0.5*1 + (-0.5)*1 + 0.2 = 0.2
    [InlineData(new double[] { 1, 1 }, 0, new double[] { 0.5, 0.5 }, 1)] // 1*0.5 + 1*0.5 + 0 = 1
    public void Calculate_ShouldReturnCorrectValue_ForLinearActivation(
        double[] weights,
        double bias,
        double[] inputs,
        double expected
    )
    {
        // Arrange
        var neuron = new Neuron(weights, bias, ActivationFunctions.FunctionType.Linear);

        // Act
        var result = neuron.Calculate(inputs);

        // Assert
        Assert.Equal(expected, result, 5);
    }

    [Fact]
    public void Calculate_ShouldReturnCorrectValue_ForSigmoidActivation()
    {
        // Arrange
        var neuron = new Neuron([0.5, 0.5], 0, ActivationFunctions.FunctionType.Sigmoid);
        var inputs = new double[] { 1, 1 };
        // sum = 1*0.5 + 1*0.5 + 0 = 1
        // expected = 1 / (1 + exp(-1)) = 0.73105...
        var expected = 0.73105857863;

        // Act
        var result = neuron.Calculate(inputs);

        // Assert
        Assert.Equal(expected, result, 5);
    }

    [Fact]
    public void Constructor_WithRandom_ShouldInitializeWeightsAndBias()
    {
        // Arrange
        var random = new Random(123); // Use a seed for predictability
        var expectedWeight1 = random.NextDouble() * 2 - 1;
        var expectedWeight2 = random.NextDouble() * 2 - 1;
        var expectedBias = random.NextDouble() * 2 - 1;

        // Act
        random = new Random(123); // Reset seed
        var neuron = new Neuron(2, random);

        // Assert
        Assert.Equal(expectedWeight1, neuron.Weights[0]);
        Assert.Equal(expectedWeight2, neuron.Weights[1]);
        Assert.Equal(expectedBias, neuron.Bias);
    }

    [Fact]
    public void CalculateSetDelta_ForOutputNeuron_ShouldSetCorrectDelta()
    {
        // Arrange
        var neuron = new Neuron([1], 0, ActivationFunctions.FunctionType.Sigmoid)
        {
            Value = 0.7, // Assume this is the output after feedforward
        };
        var target = 1.0;
        // error = 1.0 - 0.7 = 0.3
        // derivative = value * (1 - value) = 0.7 * (1 - 0.7) = 0.7 * 0.3 = 0.21
        // delta = error * derivative = 0.3 * 0.21 = 0.063
        var expectedDelta = 0.063;

        // Act
        neuron.CalculateSetDelta(target);

        // Assert
        Assert.Equal(expectedDelta, neuron.Delta, 5);
    }

    [Fact]
    public void CalculateSetDelta_ForHiddenNeuron_ShouldSetCorrectDelta()
    {
        // Arrange
        // Neuron in the hidden layer we are testing
        var hiddenNeuron = new Neuron([1], 0, ActivationFunctions.FunctionType.Sigmoid)
        {
            Value = 0.6, // Assume this is the output after feedforward
        };
        var hiddenNeuronIndex = 0;

        // The next layer (output layer)
        var outputNeuron1 = new Neuron([0.5], 0) { Delta = 0.1 }; // Weight from hiddenNeuron is 0.5
        var outputNeuron2 = new Neuron([-0.2], 0) { Delta = 0.2 }; // Weight from hiddenNeuron is -0.2
        var nextLayer = new Layer { Neurons = [outputNeuron1, outputNeuron2] };

        // error = sum(weight * delta) = (0.5 * 0.1) + (-0.2 * 0.2) = 0.05 - 0.04 = 0.01
        // derivative = value * (1 - value) = 0.6 * (1 - 0.6) = 0.6 * 0.4 = 0.24
        // delta = error * derivative = 0.01 * 0.24 = 0.0024
        var expectedDelta = 0.0024;

        // Act
        hiddenNeuron.CalculateSetDelta(nextLayer, hiddenNeuronIndex);

        // Assert
        Assert.Equal(expectedDelta, hiddenNeuron.Delta, 5);
    }
}
