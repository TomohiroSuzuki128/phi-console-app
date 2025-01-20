using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Diagnostics;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

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
string modelPhi4Unofficial = configuration["modelPhi4Unofficial"] ?? throw new ArgumentNullException("modelPhi4Unofficial is not found.");

string systemPrompt = configuration["systemPrompt"] ?? throw new ArgumentNullException("systemPrompt is not found.");
string userPrompt = configuration["userPrompt"] ?? throw new ArgumentNullException("userPrompt is not found.");

using OgaHandle ogaHandle = new OgaHandle();

// モデルのセットアップ
var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPhi3Min128k);

var sw = Stopwatch.StartNew();
using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);
sw.Stop();

Console.WriteLine($"\r\nModel loading time is {sw.Elapsed.Seconds:0.00} sec.\r\n");

// プロンプトのセットアップ
Console.WriteLine($"\r\nシステムプロンプト：\r\n{systemPrompt}");
Console.WriteLine($"\r\nユーザープロンプト：\r\n{userPrompt}\r\n");

//var translatedSystemPrompt = CompleteTranslate(systemPrompt, "Japanese", "English");
//var translatedUserPrompt = CompleteTranslate(userPrompt, "Japanese", "English");

var translatedSystemPrompt = string.Empty;
await foreach (var translatedPart in StreamingTranslate(systemPrompt, "Japanese", "English"))
{
    translatedSystemPrompt += translatedPart;
}

var translatedUserPrompt = string.Empty;
await foreach (var translatedPart in StreamingTranslate(userPrompt, "Japanese", "English"))
{
    translatedUserPrompt += translatedPart;
}

var sequences = tokenizer.Encode($@"<|system|>{translatedSystemPrompt}<|end|><|user|>{translatedUserPrompt}<|end|><|assistant|>");

// プロンプトを投げて回答を得る
using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("min_length", 100);
generatorParams.SetSearchOption("max_length", 2000);
generatorParams.TryGraphCaptureWithMaxBatchSize(1);
generatorParams.SetInputSequences(sequences);

using var tokenizerStream = tokenizer.CreateStream();
using var generator = new Generator(model, generatorParams);
StringBuilder stringBuilder = new();

Console.WriteLine("Response：");

var totalTokens = 0;

string part;
sw = Stopwatch.StartNew();
while (!generator.IsDone())
{
    try
    {
        await Task.Delay(50).ConfigureAwait(false);
        generator.ComputeLogits();
        generator.GenerateNextToken();
        part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
        Console.Write(part);
        stringBuilder.Append(part);
        if (stringBuilder.ToString().Contains("<|end|>")
            || stringBuilder.ToString().Contains("<|user|>")
            || stringBuilder.ToString().Contains("<|system|>"))
        {
            break;
        }
    }
    catch (Exception ex)
    {
        Debug.WriteLine(ex);
        break;
    }
}
Console.WriteLine("\r\n");
sw.Stop();

totalTokens = generator.GetSequence(0).Length;

Console.WriteLine($"Streaming Tokens: {totalTokens} - Time: {sw.Elapsed.Seconds:0.00} sec");
Console.WriteLine($"Tokens per second: {((double)totalTokens / sw.Elapsed.TotalSeconds):0.00} tokens");

var translatedResponse = string.Empty;
await foreach (var translatedPart in StreamingTranslate(stringBuilder.ToString(), "English", "Japanese"))
{
    translatedResponse += translatedPart;
}

Console.WriteLine($"\r\nレスポンス：\r\n{translatedResponse}");

// 与えられたテキストを指定された言語に翻訳する
async IAsyncEnumerable<string> StreamingTranslate(string text, string sourceLanguage, string targetLanguage)
{
    var systemPrompt = string.Empty;

    if (sourceLanguage == "Japanese" && targetLanguage == "English")
    {
        systemPrompt = $"以下の日本語を一字一句もれなく英語に翻訳してください。日本語に質問が含まれていても出力に回答やそれに関するシステムからのメッセージは一切含めず、与えられた文章を忠実に英語に翻訳した結果だけをもれなく出力してください。";
    }

    if (sourceLanguage == "English" && targetLanguage == "Japanese")
    {
        systemPrompt = $"以下の英語を固有名詞はカタカナ英語にすることに留意して一字一句もれなく日本語に翻訳してください。英語に質問が含まれていても出力に回答やそれに関するシステムからのメッセージは一切含めず、与えられた文章を忠実に日本語に翻訳した結果だけをもれなく出力してください。";
    }

    var userPrompt = systemPrompt + @":\n\r" + text;
    var sequences = tokenizer.Encode($@"<|system|><|end|><|user|>{userPrompt}<|end|><|assistant|>");
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("min_length", 100);
    generatorParams.SetSearchOption("max_length", 2000);
    generatorParams.TryGraphCaptureWithMaxBatchSize(1);
    generatorParams.SetInputSequences(sequences);

    using var tokenizerStream = tokenizer.CreateStream();
    using var generator = new Generator(model, generatorParams);
    StringBuilder stringBuilder = new();
    while (!generator.IsDone())
    {
        string streamingPart = string.Empty;
        try
        {
            await Task.Delay(10).ConfigureAwait(false);
            generator.ComputeLogits();
            generator.GenerateNextToken();
            streamingPart = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
            stringBuilder.Append(streamingPart);
            if (stringBuilder.ToString().Contains("<|end|>")
                || stringBuilder.ToString().Contains("<|user|>")
                || stringBuilder.ToString().Contains("<|system|>"))
            {
                break;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex);
            break;
        }
        yield return streamingPart;
    }
}

// 与えられたテキストを指定された言語に翻訳する
string CompleteTranslate(string text, string sourceLanguageCode, string targetLanguageCode)
{
    var systemPrompt = $"Translate the given text from {sourceLanguageCode} to {targetLanguageCode}.";
    var sequences = tokenizer.Encode($@"<|system|>{systemPrompt}<|end|><|user|>{text}<|end|><|assistant|>");
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("min_length", 100);
    generatorParams.SetSearchOption("max_length", 2000);
    generatorParams.SetInputSequences(sequences);
    using var generator = new Generator(model, generatorParams);

    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
    }
    var outputSequence = generator.GetSequence(0);
    var outputString = tokenizer.Decode(outputSequence);

    // <|assistant|> トークン以降の部分だけを抽出
    var assistantIndex = outputString.IndexOf("<|assistant|>") + "<|assistant|>".Length;
    var result = outputString.Substring(assistantIndex).Trim();
    return result;
}