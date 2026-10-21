using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;


class Tokenizer
{
    public Dictionary<int, string> slovnik = new Dictionary<int, string>();//slovník
    public List<int> TokenizedText = new List<int>();

    // znak pro mezeru
    private const string MezeraToken = "Ġ";
    private const string SouborSlovniku = "slovnik.json";

    public Tokenizer()
    {
        
    }

    void TokenizerStart(string input)
    {
        Load();
        string textToTokenize = Console.ReadLine();
        string upravenyText = textToTokenize.ToLower().Replace(" ", MezeraToken);
        if (slovnik.Count == 0)
        {
            TrainBPE(upravenyText);
            Save();
        }

        Tokenize(upravenyText);
        
        string rekonstrukce = Detokenize(TokenizedText);
    }
    
    public void TrainBPE(string textToTokenize) //BPE - trénink
    {
        foreach (char c in textToTokenize)
        {
            string znak = c.ToString();
            if (!slovnik.ContainsValue(znak))
            {
                slovnik.Add(slovnik.Count, znak);
            }
        }
        
        while (true)
        {
            List<int> aktualniTokeny = SlovnikNaText(textToTokenize);
            
            var frekvenceDvojic = new Dictionary<(int, int), int>();

            for (int i = 0; i < aktualniTokeny.Count - 1; i++)
            {
                var dvojice = (aktualniTokeny[i], aktualniTokeny[i + 1]);
                if (frekvenceDvojic.ContainsKey(dvojice))
                    frekvenceDvojic[dvojice]++;
                else
                    frekvenceDvojic[dvojice] = 1;
            }

            if (frekvenceDvojic.Count == 0) break;
            
            var nejcastejsi = frekvenceDvojic.OrderByDescending(x => x.Value).First();
            
            if (nejcastejsi.Value <= 1)
            {
                break;
            }
            
            string value1 = slovnik[nejcastejsi.Key.Item1];
            string value2 = slovnik[nejcastejsi.Key.Item2];
            string novyToken = value1 + value2;

            slovnik.Add(slovnik.Count, novyToken);
        }
    }
    
    public void Tokenize(string text)
    {
        TokenizedText = SlovnikNaText(text);
    }
    
    private List<int> SlovnikNaText(string text)
    {
        List<int> vysledek = new List<int>();
        int i = 0;
        while (i < text.Length)
        {
            string nejdelsiNajdenyToken = "";
            int idNejdelsiho = -1;
            
            foreach (var token in slovnik)
            {
                if (text.Substring(i).StartsWith(token.Value))
                {
                    if (token.Value.Length > nejdelsiNajdenyToken.Length)
                    {
                        nejdelsiNajdenyToken = token.Value;
                        idNejdelsiho = token.Key;
                    }
                }
            }

            if (idNejdelsiho != -1)
            {
                vysledek.Add(idNejdelsiho);
                i += nejdelsiNajdenyToken.Length;
            }
            else
            {
                i++; 
            }
        }
        return vysledek;
    }
    
    public string Detokenize(List<int> tokenIds)
    {
        string text = "";
        foreach (var id in tokenIds)
        {
            if (slovnik.ContainsKey(id))
            {
                text += slovnik[id];
            }
        }
        return text.Replace(MezeraToken, " ");
    }

    public void VypisSlovnik()
    {
        foreach (var s in slovnik.OrderBy(x => x.Key))
        {
            Console.WriteLine($"{s.Key}: \"{s.Value}\"");
        }
    }

    // ukládání
    public void Save()
    {
        try
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(slovnik, options);
            File.WriteAllText(SouborSlovniku, jsonString);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Chyba při ukládání slovníku: {ex.Message}");
        }
    }

    // načítání
    public void Load()
    {
        if (File.Exists(SouborSlovniku))
        {
            try
            {
                string jsonString = File.ReadAllText(SouborSlovniku);
                var sSlovnik = JsonSerializer.Deserialize<Dictionary<string, string>>(jsonString);
                
                slovnik = sSlovnik.ToDictionary(x => int.Parse(x.Key), x => x.Value);
            }
            catch (Exception ex)
            {
                slovnik.Clear();
            }
        }
    }
}
