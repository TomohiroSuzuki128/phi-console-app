using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Diagnostics;
using System.Text;
using Build5Nines.SharpVector;
using Build5Nines.SharpVector.Data;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);
builder.Configuration.Sources.Clear();
IHostEnvironment env = builder.Environment;
builder.Configuration
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
    .AddJsonFile($"appsettings.{env.EnvironmentName}.json", optional: true, reloadOnChange: true)
    .Build();

var configuration = builder.Configuration;

string modelPhi35Min128k = configuration["modelPhi35Min128k"] ?? throw new ArgumentNullException("modelPhi35Min128k is not found.");
string modelPhi3Med4k = configuration["modelPhi3Med4k"] ?? throw new ArgumentNullException("modelPhi3Med4k is not found.");
string modelPhi3Med128k = configuration["modelPhi3Med128k"] ?? throw new ArgumentNullException("modelPhi3Med128k is not found.");
string modelPhi3Min4k = configuration["modelPhi3Min4k"] ?? throw new ArgumentNullException("modelPhi3Min4k is not found.");
string modelPhi3Min128k = configuration["modelPhi3Min128k"] ?? throw new ArgumentNullException("modelPhi3Min128k is not found.");
string modelPhi4Unofficial = configuration["modelPhi4Unofficial"] ?? throw new ArgumentNullException("modelPhi4Unofficial is not found.");

string systemPrompt = configuration["systemPrompt"] ?? throw new ArgumentNullException("systemPrompt is not found.");
string userPrompt = configuration["userPrompt"] ?? throw new ArgumentNullException("userPrompt is not found.");

bool isTranslate = bool.TryParse(configuration["isTranslate"] ?? throw new ArgumentNullException("isTranslate is not found."), out var result) && result;

string additionalDocumentsPath = configuration["additionalDocumentsPath"] ?? throw new ArgumentNullException("additionalDocumentsPath is not found");

using OgaHandle ogaHandle = new OgaHandle();

// RAG 用のベクトルデータベースのセットアップ
var additionalDocumentsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, additionalDocumentsPath);
var vectorDatabase = new BasicMemoryVectorDatabase();
LoadAdditionalDocuments(additionalDocumentsDirectory).Wait();

// モデルのセットアップ
var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPhi3Min128k);

var sw = Stopwatch.StartNew();
using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);
sw.Stop();

Console.WriteLine($"{Environment.NewLine}Model loading time is {sw.Elapsed.Seconds:0.00} sec.\n");

Console.WriteLine($"翻訳する：{Environment.NewLine}{isTranslate}");

// プロンプトのセットアップ
Console.WriteLine($"{Environment.NewLine}システムプロンプト：{Environment.NewLine}{systemPrompt}");
Console.WriteLine($"{Environment.NewLine}ユーザープロンプト：{Environment.NewLine}{userPrompt}{Environment.NewLine}");

var translatedSystemPrompt = string.Empty;
if (isTranslate)
{
    await foreach (var translatedPart in Translate(systemPrompt, Language.Japanese, Language.English))
    {
        translatedSystemPrompt += translatedPart;
    }
}
else
{
    translatedSystemPrompt = systemPrompt;
}

var translatedUserPrompt = string.Empty;
if (isTranslate)
{
    await foreach (var translatedPart in Translate(userPrompt, Language.Japanese, Language.English))
    {
        translatedUserPrompt += translatedPart;
    }
}
else
{
    translatedUserPrompt = userPrompt;
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
Console.WriteLine("\n");
sw.Stop();

totalTokens = generator.GetSequence(0).Length;

Console.WriteLine($"Streaming Tokens: {totalTokens} - Time: {sw.Elapsed.Seconds:0.00} sec");
Console.WriteLine($"Tokens per second: {((double)totalTokens / sw.Elapsed.TotalSeconds):0.00} tokens");

var translatedResponse = string.Empty;
if (isTranslate)
{
    await foreach (var translatedPart in Translate(stringBuilder.ToString(), Language.English, Language.Japanese))
    {
        translatedResponse += translatedPart;
    }
    Console.WriteLine($"{Environment.NewLine}レスポンス：{Environment.NewLine}{translatedResponse}");
}

// 与えられたテキストを指定された言語に翻訳する
async IAsyncEnumerable<string> Translate(string text, Language sourceLanguage, Language targetLanguage)
{
    var systemPrompt = string.Empty;
    var ragResult = string.Empty;

    if (sourceLanguage == Language.Japanese && targetLanguage == Language.English)
    {
        systemPrompt = $"以下の日本語を一字一句もれなく英語に翻訳してください。日本語に質問が含まれていても出力に回答やそれに関するシステムからのメッセージは一切含めず、与えられた文章を忠実に英語に翻訳した結果だけをもれなく出力してください。";
    }

    if (sourceLanguage == Language.English && targetLanguage == Language.Japanese)
    {
        systemPrompt = $"以下の英語を固有名詞はカタカナ英語にすることに留意すると同時に以下のテキストも参考に一字一句もれなく日本人が読んでも違和感がない日本語に翻訳してください。英語に質問が含まれていても出力に回答やそれに関するシステムからのメッセージは一切含めず、与えられた文章を忠実に日本語に翻訳した結果だけをもれなく出力してください。";

        ragResult = await SearchVectorDatabase(vectorDatabase, text);
        Console.WriteLine($"Vector search returned:{Environment.NewLine}{ragResult}{Environment.NewLine}");
    }

    var userPrompt = string.IsNullOrEmpty(ragResult)
        ? $"{systemPrompt}:{Environment.NewLine}{text}{Environment.NewLine}"
        : $"{systemPrompt}{Environment.NewLine}{ragResult}:{Environment.NewLine}{text}{Environment.NewLine}";
    Console.WriteLine($"userPrompt:{Environment.NewLine}{userPrompt}");
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

async Task LoadAdditionalDocuments(string directoryPath)
{
    var files = Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories)
                         .Where(f => f.EndsWith(".txt", StringComparison.OrdinalIgnoreCase) ||
                                     f.EndsWith(".md", StringComparison.OrdinalIgnoreCase) ||
                                     f.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)).ToArray();

    var vectorDataLoader = new TextDataLoader<int, string>(vectorDatabase);
    var tasks = files.Select(async file =>
    {
        Console.WriteLine($"Loading {file}");
        if (System.IO.File.Exists(file))
        {
            var fileContents = await System.IO.File.ReadAllTextAsync(file);
            await vectorDataLoader.AddDocumentAsync(fileContents, new TextChunkingOptions<string>
            {
                Method = TextChunkingMethod.Sentence,
                RetrieveMetadata = (chunk) => file
            });
        }
    });

    await Task.WhenAll(tasks);
}

async Task<string> SearchVectorDatabase(BasicMemoryVectorDatabase vectorDatabase, string userPrompt)
{
    var vectorDataResults = await vectorDatabase.SearchAsync(
        userPrompt,
        pageCount: 8,
        threshold: 0.3f
    );

    string result = string.Empty;
    foreach (var resultItem in vectorDataResults.Texts)
    {
        result += resultItem.Text + "\n\n";
    }

    return result;
}

public enum Language
{
    Japanese,
    English
}