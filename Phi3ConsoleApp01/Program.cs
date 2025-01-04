using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Diagnostics;


var env = Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT") ?? string.Empty;
var configuration = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json")
    .AddJsonFile($"appsettings.{env}.json", true)
    .Build();

string modelPhi35Min128k = configuration["modelPhi35Min128k"] ?? throw new ArgumentNullException("modelPhi35Min128k is not found.");
string modelPhi3Med4k = configuration["modelPhi3Med4k"] ?? throw new ArgumentNullException("modelPhi3Med4k is not found.");
string modelPhi3Med128k = configuration["modelPhi3Med128k"] ?? throw new ArgumentNullException("modelPhi3Med128k is not found.");
string modelPhi3Min4k = configuration["modelPhi3Min4k"] ?? throw new ArgumentNullException("modelPhi3Min4k is not found.");
string modelPhi3Min128k = configuration["modelPhi3Min128k"] ?? throw new ArgumentNullException("modelPhi3Min128k is not found.");

using OgaHandle ogaHandle = new OgaHandle();

// モデルのセットアップ
var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPhi35Min128k);

var sw = Stopwatch.StartNew();
using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);
sw.Stop();

Console.WriteLine($"\r\nModel loading time is {sw.Elapsed.Seconds:0.00} sec.\r\n");

sw = Stopwatch.StartNew();
// プロンプトのセットアップ
//var systemPrompt = "あなたはRPGゲーム、「ドラゴンクエスト」に詳しいゲームの達人です。与えられた質問にドラゴンクエストの知識を最大限活用して解説してください。";
//var userPrompt = "「ドラゴンクエスト」のキャラクターデザインを担当したのは誰か教えてください。";

var systemPrompt = "あなたはRPGゲーム、「ファイナルファンタジー7」に詳しいゲームの達人です。与えられた質問にファイナルファンタジー7の知識を最大限活用して解説してください。";
var userPrompt = "「ファイナルファンタジー7」の主人公の名前を教えてください。";

//var systemPrompt = "You are a game guru who is familiar with the RPG game, Final Fantasy 7. Please explain the given question by making the best use of your knowledge of Final Fantasy 7.";
//var userPrompt = "What is the name of the main character in Final Fantasy 7?";

Console.WriteLine($"\r\n{userPrompt}\r\n");

var sequences = tokenizer.Encode($@"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>");

using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("min_length", 100);
generatorParams.SetSearchOption("max_length", 2000);
generatorParams.TryGraphCaptureWithMaxBatchSize(1);
generatorParams.SetInputSequences(sequences);

using var tokenizerStream = tokenizer.CreateStream();
using var generator = new Generator(model, generatorParams);

var totalTokens = 0;
while (!generator.IsDone())
{
    try
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
    }
    catch (Exception ex)
    {
        Debug.WriteLine(ex);
        break;
    }
}
sw.Stop();
Console.WriteLine("\r\n");

totalTokens = generator.GetSequence(0).Length;

Console.WriteLine($"Streaming Tokens: {totalTokens} - Time: {sw.Elapsed.Seconds:0.00} sec");
Console.WriteLine($"Tokens per second: {(totalTokens / sw.Elapsed.Seconds):0.000} tokens");