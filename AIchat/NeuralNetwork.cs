using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.IO;

namespace Neurony
{
    class NeuralNetwork
    {
        public Layer[] Layers;

        public NeuralNetwork(int[] layerSizes)
        {
            Layers = new Layer[layerSizes.Length - 1];

            for (int i = 0; i < Layers.Length; i++)
            {
                int inputCount = layerSizes[i];
                int neuronCount = layerSizes[i + 1];
                bool isOutput = (i == Layers.Length - 1);

                Layers[i] = new Layer(inputCount, neuronCount, isOutput);
            }
        }

        public double[] Predict(double[] inputs)
        {
            double[] outputs = inputs;

            for (int i = 0; i < Layers.Length; i++)
            {
                outputs = Layers[i].Predict(outputs);
            }

            return outputs;
        }

        public void Train(double[] inputs, double[] targets)
        {
            double[][] layerInputs = new double[Layers.Length][];
            double[] currentInput = inputs;

            for (int i = 0; i < Layers.Length; i++)
            {
                layerInputs[i] = currentInput;
                currentInput = Layers[i].Predict(currentInput);
            }

            Layer outputLayer = Layers[Layers.Length - 1];
            outputLayer.CalculateOutputGradients(targets);

            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                Layers[i].CalculateHiddenGradients(Layers[i + 1]);
            }

            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].UpdateWeights(layerInputs[i]);
            }
        }

        public void Save(string filePath)
        {
            var options = new JsonSerializerOptions { IncludeFields = true, WriteIndented = true };
            string json = JsonSerializer.Serialize(this.Layers, options);
            File.WriteAllText(filePath, json);
            Console.WriteLine("Váhy byly uloženy.");
        }

        public void Load(string filePath)
        {
            if (File.Exists(filePath))
            {
                var options = new JsonSerializerOptions { IncludeFields = true };
                string json = File.ReadAllText(filePath);
                this.Layers = JsonSerializer.Deserialize<Layer[]>(json, options);
                Console.WriteLine("Váhy byly načteny.");
            }
        }
    }
}