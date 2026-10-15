class Neuron
{
    public double[] Weights;
    public double Bias;
    public double LearningRate;
    public double LastOutput;
    public double Gradient;

    private static Random random = new Random();

    public Neuron() { }

    public Neuron(int inputCount, double learningRate = 0.005)
    {
        Weights = new double[inputCount];
        LearningRate = learningRate;
        
        double limit = Math.Sqrt(2.0 / inputCount);

        for (int i = 0; i < inputCount; i++)
        {
            Weights[i] = (random.NextDouble() * 2 - 1) * limit;
        }

        Bias = 0.0;
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    
    public double PredictReLU(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        LastOutput = sum > 0 ? sum : 0;
        return LastOutput;
    }
    
    public double PredictSigmoid(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        LastOutput = Sigmoid(sum);
        return LastOutput;
    }
    
    public void CalculateOutputGradient(double target)
    {
        Gradient = (LastOutput - target) * LastOutput * (1.0 - LastOutput);
    }
    
    public void CalculateHiddenGradientReLU(double nextLayerSum)
    {
        Gradient = LastOutput > 0 ? nextLayerSum : 0;
    }

    public void UpdateWeights(double[] inputs)
    {
        for (int i = 0; i < Weights.Length; ++i)
        {
            Weights[i] -= LearningRate * Gradient * inputs[i];
        }
        Bias -= LearningRate * Gradient;
    }
}