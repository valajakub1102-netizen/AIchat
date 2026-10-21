using System;
using Neurony;

class Program
{
    public static NeuralNetwork NeuralNetwork = new NeuralNetwork(new int[] {3000, 256, 128, 1000});
    public static Tokenizer Tokenizer = new Tokenizer();
    public static void Main()
    {
        string inp = Console.ReadLine();
        if (inp == "admin1234") Train(); else Talk();
    }

    public static void Train()
    {
        Console.Clear();
        Console.WriteLine("-------AI-CHATBOT-train-------");
        Console.WriteLine("1. AI 2. Tokenizer");
        string input = Console.ReadLine();
        if (input == "1") 
        {
            //from ai.txt přes tokenizer do neuronové sítě ("You>..." - řeč od user, "AI>..." - ai odpověď)
        } else if (input == "2")
        {
            //from token.txt na vytvoření tokenizeru
        }
    }

    public static void Talk()
    {
        Console.Clear();
        Console.WriteLine("-------------------------AI-CHATBOT-------------------------");
        while (true)
        {
            Console.Write("> YOU: ");
            string input = Console.ReadLine();
            Tokenizer.Tokenize(input);
            double[] response = NeuralNetwork.Predict(Tokenizer.TokenizedText.ConvertAll(x => (double)x).ToArray());
            string textResponse = Tokenizer.Detokenize(response.Select(x => (int)x).ToList());
            Console.WriteLine($"> AI: {textResponse}");
        }
    }
}