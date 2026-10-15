namespace Neurony
{
    class Layer
    {
        public Neuron[] Neurons;
        public bool IsOutputLayer;

        public Layer() { }

        public Layer(int inputCount, int neuronCount, bool isOutputLayer = false)
        {
            IsOutputLayer = isOutputLayer;
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(inputCount);
            }
        }

        public double[] Predict(double[] inputs)
        {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                outputs[i] = IsOutputLayer 
                    ? Neurons[i].PredictSigmoid(inputs) 
                    : Neurons[i].PredictReLU(inputs);
            }
            return outputs;
        }

        public void CalculateOutputGradients(double[] targets)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].CalculateOutputGradient(targets[i]);
            }
        }

        public void CalculateHiddenGradients(Layer nextLayer)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < nextLayer.Neurons.Length; j++)
                {
                    sum += nextLayer.Neurons[j].Gradient * nextLayer.Neurons[j].Weights[i];
                }
                Neurons[i].CalculateHiddenGradientReLU(sum);
            }
        }

        public void UpdateWeights(double[] inputs)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].UpdateWeights(inputs);
            }
        }
    }
}