# simple-perceptron

A simple, highly configurable perceptron created using C# and .NET 9.

Features:

- Configurable network structure (user-defined input, hidden, and output layer size)
- Configurable activation functions (defined per-layer)
- Backpropagation training (with customizable learning rates and epochs)
- Hyperparameter tuning program
- Model persistence (complete state import/export of a trained perceptron)

## Structure

The `Perceptron` class is the main class representing the neural network. The perceptron is made of connected `Layer`s, which are each a collection of `Neuron`s. Each `Neuron` is connected to every `Neuron` in the previous `Layer`, and when predicting, uses the configured `ActivationFunction` to transform the set of inputs into an output (as a double), which it then passes to every `Neuron` in the next layer. A similar process is performed in reverse during backpropogation, where the internal weights of each `Neuron` are modified to better fit the expected output.

The `PerceptronTuner` is a utility class for hyperparameter optimization using a genetic algorithm. 

## Usage

use it :)
