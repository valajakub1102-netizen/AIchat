using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Neurony;

class Program
{
    public static GpuNeuralNetwork NeuralNetwork = null!;
    public static Tokenizer Tokenizer = new Tokenizer();

    private const int InputSize = 3000;
    private const int Hidden1 = 256;
    private const int Hidden2 = 128;
    private const int Epochs = 100;
    private const int MaxResponseTokens = 100;

    
    public static void Main()
    {
        if (File.Exists("slovnik.json"))
            Tokenizer.Load("slovnik.json");

        InitializeNetwork();

        Console.Write("Přístupový kód: ");
        string input = Console.ReadLine() ?? "";

        if (input == "admin1234")
            Train();
        else
            Talk();
    }


    public static void InitializeNetwork()
    {
        int outputSize = Tokenizer.slovnik.Count == 0
            ? 1
            : Tokenizer.slovnik.Keys.Max() + 1;

        NeuralNetwork?.Dispose();
        NeuralNetwork = new GpuNeuralNetwork(
            new[] { InputSize, Hidden1, Hidden2, outputSize });

        if (File.Exists("vahy.bin"))
        {
            try
            {
                NeuralNetwork.Load("vahy.bin");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Model se musí znovu natrénovat: {ex.Message}");
            }
        }
    }

    public static void Train()
    {
        Console.Clear();
        Console.WriteLine("-------AI-CHATBOT-train-------");
        Console.WriteLine("1. AI  2. Tokenizer");

        string input = Console.ReadLine() ?? "";

        if (input == "1")
            TrainAI();
        else if (input == "2")
            TrainTokenizer();
    }

    public static float[] EncodeContext(List<int> tokenIds)
    {
        float[] result = new float[InputSize];
        int start = Math.Max(0, tokenIds.Count - InputSize);
        int count = tokenIds.Count - start;
        int vocabularySize = Math.Max(1, Tokenizer.slovnik.Count);

        for (int i = 0; i < count; i++)
            result[i] = (tokenIds[start + i] + 1f) / vocabularySize;

        return result;
    }

    public static int PredictNextToken(List<int> context)
    {
        float[] output = NeuralNetwork.Predict(EncodeContext(context));

        int bestId = 0;
        float bestValue = float.NegativeInfinity;

        for (int i = 0; i < output.Length; i++)
        {
            if (float.IsFinite(output[i]) && output[i] > bestValue)
            {
                bestValue = output[i];
                bestId = i;
            }
        }

        return bestId;
    }

    public static void Talk()
    {
        Console.Clear();
        Console.WriteLine("----------------AI-CHATBOT----------------");

        if (Tokenizer.EndTokenId < 0 || Tokenizer.slovnik.Count == 0)
        {
            Console.WriteLine("Nejdřív natrénuj tokenizer.");
            return;
        }

        while (true)
        {
            Console.Write("> YOU: ");
            string input = Console.ReadLine() ?? "";

            if (string.IsNullOrWhiteSpace(input))
                continue;

            try
            {
                Tokenizer.Tokenize(input);
                List<int> context = new List<int>(Tokenizer.TokenizedText);
                List<int> response = new List<int>();

                for (int i = 0; i < MaxResponseTokens; i++)
                {
                    int nextToken = PredictNextToken(context);

                    if (nextToken == Tokenizer.EndTokenId)
                        break;

                    if (!Tokenizer.slovnik.ContainsKey(nextToken))
                        break;

                    response.Add(nextToken);
                    context.Add(nextToken);
                }

                string textResponse = Tokenizer.Detokenize(response);
                Console.WriteLine($"> AI: {textResponse}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Chyba: {ex.Message}");
            }
        }
    }

    public static void TrainAI()
    {
        string pathAI = "dataset_cs.txt"; //učební soubor
        Console.Clear();
        Console.WriteLine("-------AI-CHATBOT-train-AI-------");

        if (!File.Exists(pathAI))
        {
            Console.WriteLine( pathAI + " not found!");
            return;
        }

        if (Tokenizer.EndTokenId < 0)
        {
            Console.WriteLine("Nejdřív natrénuj tokenizer.");
            return;
        }

        for (int i = 0; i < 500; i++)
        {
            List<(string Input, string Output)> pairs = new();
            string? userInput = null;

            foreach (string line in File.ReadAllLines(pathAI))
            {
                if (line.StartsWith("You>"))
                {
                    userInput = line.Substring(4).Trim();
                }
                else if (line.StartsWith("AI>") && userInput != null)
                {
                    string answer = line.Substring(3).Trim();

                    if (!string.IsNullOrWhiteSpace(userInput) &&
                        !string.IsNullOrWhiteSpace(answer))
                        pairs.Add((userInput, answer));

                    userInput = null;
                }
            }

            if (pairs.Count == 0)
            {
                Console.WriteLine("Dataset neobsahuje žádné platné dvojice.");
                return;
            }

            foreach (var pair in pairs)
            {
                Tokenizer.Tokenize(pair.Input);
                List<int> inputTokens = new(Tokenizer.TokenizedText);

                Tokenizer.Tokenize(pair.Output);
                List<int> targetTokens = new(Tokenizer.TokenizedText);
                targetTokens.Add(Tokenizer.EndTokenId);

                if (inputTokens.Count == 0)
                    continue;

                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    List<int> context = new(inputTokens);

                    foreach (int targetToken in targetTokens)
                    {
                        if (targetToken < 0 ||
                            targetToken >= NeuralNetwork.OutputSize)
                            continue;

                        NeuralNetwork.Train(EncodeContext(context), targetToken);
                        context.Add(targetToken);
                    }
                }

                Console.WriteLine($"Natrénováno [{i}] : {pair.Input}");
            }


            NeuralNetwork.Save("vahy.bin");
            Console.WriteLine($"Trénování dokončeno. Dvojic: {pairs.Count}");
        }
    }

    
    public static void TrainTokenizer()
    {
        Console.Clear();
        Console.WriteLine("-------AI-CHATBOT-tokenizer-------");

        if (!File.Exists("token.txt"))
        {
            Console.WriteLine("Soubor token.txt nebyl nalezen.");
            Console.ReadKey();
            return;
        }

        Tokenizer.TrainBPE(File.ReadAllText("token.txt"));
        Tokenizer.EnsureEndToken();

        if (Tokenizer.EndTokenId < 0)
        {
            Console.WriteLine("Nepodařilo se vytvořit END token.");
            Console.ReadKey();
            return;
        }

        if (File.Exists("vahy.bin"))
            File.Delete("vahy.bin");

        InitializeNetwork();

        Console.WriteLine("Tokenizer byl úspěšně natrénován.");
        Console.WriteLine($"Počet tokenů: {Tokenizer.slovnik.Count}");
        Console.WriteLine($"END token ID: {Tokenizer.EndTokenId}");
        Console.WriteLine("Neuronová síť byla resetována. Teď natrénuj AI.");

        Console.ReadKey();
    }

}