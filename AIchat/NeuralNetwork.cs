
using System;
using System.IO;
using System.Text.Json;

namespace Neurony
{
    public class NeuralNetwork
    {
        public Layer[] Layers = Array.Empty<Layer>();

        public NeuralNetwork() { }

        public NeuralNetwork(int[] layerSizes)
        {
            Layers = new Layer[layerSizes.Length - 1];

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(
                    layerSizes[i],
                    layerSizes[i + 1],
                    i == Layers.Length - 1
                );
            }
        }

        public double[] Predict(double[] inputs)
        {
            double[] outputs = inputs;

            foreach (Layer layer in Layers)
                outputs = layer.Predict(outputs);

            return outputs;
        }

        public void Train(double[] inputs, double[] targets)
        {
            if (inputs.Length != Layers[0].Neurons[0].Weights.Length)
                throw new ArgumentException("Input size does not match network.");

            if (targets.Length != Layers[^1].Neurons.Length)
                throw new ArgumentException("Target size does not match network.");

            double[][] layerInputs = new double[Layers.Length][];
            double[] currentInput = inputs;

            for (int i = 0; i < Layers.Length; i++)
            {
                layerInputs[i] = currentInput;
                currentInput = Layers[i].Predict(currentInput);
            }

            Layers[^1].CalculateOutputGradients(targets);

            for (int i = Layers.Length - 2; i >= 0; i--)
                Layers[i].CalculateHiddenGradients(Layers[i + 1]);

            for (int i = 0; i < Layers.Length; i++)
                Layers[i].UpdateWeights(layerInputs[i]);
        }

        public void Save(string filePath)
        {
            var options = new JsonSerializerOptions
            {
                IncludeFields = true,
                WriteIndented = true
            };

            File.WriteAllText(filePath, JsonSerializer.Serialize(Layers, options));
        }

        public void Load(string filePath)
        {
            var options = new JsonSerializerOptions
            {
                IncludeFields = true
            };

            Layer[]? loaded = JsonSerializer.Deserialize<Layer[]>(
                File.ReadAllText(filePath), options);

            if (loaded == null || loaded.Length != Layers.Length)
                throw new InvalidDataException("Invalid network file.");

            for (int i = 0; i < loaded.Length; i++)
            {
                if (loaded[i].Neurons.Length != Layers[i].Neurons.Length)
                    throw new InvalidDataException("Network architecture mismatch.");

                for (int j = 0; j < loaded[i].Neurons.Length; j++)
                {
                    if (loaded[i].Neurons[j].Weights.Length !=
                        Layers[i].Neurons[j].Weights.Length)
                        throw new InvalidDataException("Network input size mismatch.");
                }
            }

            Layers = loaded;
        }
    }
}
