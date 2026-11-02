
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

public class Tokenizer
{
    public Dictionary<int, string> slovnik = new();
    public List<int> TokenizedText = new();
    public List<MergeRule> MergeRules = new();

    private readonly Dictionary<string, int> tokenToId = new();

    private const string MezeraToken = "\u0120";
    private const string EndToken = "<|end|>";

    public int EndTokenId { get; private set; } = -1;

    public class MergeRule
    {
        public string Left { get; set; } = "";
        public string Right { get; set; } = "";
        public string Result { get; set; } = "";
    }

    private class ModelData
    {
        public Dictionary<int, string> Vocabulary { get; set; } = new();
        public List<MergeRule> Merges { get; set; } = new();
    }

    private int AddToken(string token)
    {
        if (tokenToId.TryGetValue(token, out int existingId))
            return existingId;

        int id = slovnik.Count == 0 ? 0 : slovnik.Keys.Max() + 1;

        slovnik[id] = token;
        tokenToId[token] = id;

        return id;
    }

    private void RebuildTokenIndex()
    {
        tokenToId.Clear();

        foreach (var item in slovnik.OrderBy(x => x.Key))
        {
            if (!tokenToId.ContainsKey(item.Value))
                tokenToId[item.Value] = item.Key;
        }
    }

    public void EnsureEndToken()
    {
        if (tokenToId.TryGetValue(EndToken, out int existingId))
        {
            EndTokenId = existingId;
            return;
        }

        EndTokenId = AddToken(EndToken);
        Save();
    }

    public void TrainBPE(string text, int maxMerges = 500)
    {
        slovnik.Clear();
        TokenizedText.Clear();
        MergeRules.Clear();
        tokenToId.Clear();
        EndTokenId = -1;

        text = Normalize(text);

        foreach (char c in text)
            AddToken(c.ToString());

        List<string> symbols = text.Select(c => c.ToString()).ToList();

        for (int merge = 0; merge < maxMerges; merge++)
        {
            Dictionary<(string Left, string Right), int> pairCounts = new();

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);

                if (!pairCounts.ContainsKey(pair))
                    pairCounts[pair] = 0;

                pairCounts[pair]++;
            }

            if (pairCounts.Count == 0)
                break;

            var bestPair = pairCounts.OrderByDescending(x => x.Value).First();

            if (bestPair.Value < 2)
                break;

            string left = bestPair.Key.Left;
            string right = bestPair.Key.Right;
            string result = left + right;

            MergeRules.Add(new MergeRule
            {
                Left = left,
                Right = right,
                Result = result
            });

            AddToken(result);

            List<string> mergedSymbols = new();
            int index = 0;

            while (index < symbols.Count)
            {
                if (index < symbols.Count - 1 &&
                    symbols[index] == left &&
                    symbols[index + 1] == right)
                {
                    mergedSymbols.Add(result);
                    index += 2;
                }
                else
                {
                    mergedSymbols.Add(symbols[index]);
                    index++;
                }
            }

            symbols = mergedSymbols;
        }

        RebuildTokenIndex();
        Save();
    }

    public void Tokenize(string text)
    {
        TokenizedText.Clear();

        text = Normalize(text);

        List<string> symbols = text.Select(c => c.ToString()).ToList();

        foreach (MergeRule rule in MergeRules)
        {
            List<string> mergedSymbols = new();
            int index = 0;

            while (index < symbols.Count)
            {
                if (index < symbols.Count - 1 &&
                    symbols[index] == rule.Left &&
                    symbols[index + 1] == rule.Right)
                {
                    mergedSymbols.Add(rule.Result);
                    index += 2;
                }
                else
                {
                    mergedSymbols.Add(symbols[index]);
                    index++;
                }
            }

            symbols = mergedSymbols;
        }

        foreach (string symbol in symbols)
        {
            if (!tokenToId.TryGetValue(symbol, out int id))
                throw new InvalidOperationException(
                    $"Token '{symbol}' není ve slovníku. Přidej ho do trénovacích dat tokenizeru a tokenizer znovu natrénuj.");

            TokenizedText.Add(id);
        }
    }

    public string Detokenize(List<int> tokens)
    {
        List<string> symbols = new();

        foreach (int id in tokens)
        {
            if (id == EndTokenId)
                break;

            if (slovnik.TryGetValue(id, out string? token))
                symbols.Add(token);
        }

        return Denormalize(string.Concat(symbols));
    }

    private string Normalize(string text)
    {
        return text.Replace(" ", MezeraToken);
    }

    private string Denormalize(string text)
    {
        return text.Replace(MezeraToken, " ");
    }

    public void Save(string filePath = "slovnik.json")
    {
        ModelData data = new()
        {
            Vocabulary = slovnik,
            Merges = MergeRules
        };

        JsonSerializerOptions options = new()
        {
            WriteIndented = true
        };

        string json = JsonSerializer.Serialize(data, options);
        File.WriteAllText(filePath, json);
    }

    public void Load(string filePath = "slovnik.json")
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Soubor tokenizeru nebyl nalezen.", filePath);

        string json = File.ReadAllText(filePath);

        ModelData? data = JsonSerializer.Deserialize<ModelData>(json);

        if (data == null)
            throw new InvalidOperationException("Soubor tokenizeru se nepodařilo načíst.");

        slovnik = data.Vocabulary ?? new Dictionary<int, string>();
        MergeRules = data.Merges ?? new List<MergeRule>();

        TokenizedText.Clear();
        RebuildTokenIndex();

        EndTokenId = tokenToId.TryGetValue(EndToken, out int id)
            ? id
            : -1;
    }
}
