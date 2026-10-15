namespace Neurony;

public class Tokenizer
{
    Dictionary<string, int> wordToId = new Dictionary<string, int>();
    Dictionary<int, string> idToWord = new Dictionary<int, string>();
    private int nextId = 0;

    public Tokenizer()
    {
        AddToken("<PAD>");
        AddToken("UNK");
    }

    public void AddToken(string word)
    {
        if (!wordToId.ContainsKey(word))
        {
            wordToId[word] = nextId;
            idToWord[nextId] = word;
            nextId++;
        }
    }
}