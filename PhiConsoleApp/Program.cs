using Microsoft.ML.OnnxRuntimeGenAI;
using System.Diagnostics;
using System.Text;
using Build5Nines.SharpVector;
using Build5Nines.SharpVector.Data;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;

var newLine = Environment.NewLine;

var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.Sources.Clear();
builder.Configuration
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
    .AddJsonFile($"appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)
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

bool isTranslate = bool.TryParse(configuration["isTranslate"] ?? throw new ArgumentNullException("isTranslate is not found."), out var resultIsTranslate) && resultIsTranslate;
bool isUsingRag = bool.TryParse(configuration["isUsingRag"] ?? throw new ArgumentNullException("isUsingRag is not found."), out var resultIsUsingRag) && resultIsUsingRag;

string additionalDocumentsPath = configuration["additionalDocumentsPath"] ?? throw new ArgumentNullException("additionalDocumentsPath is not found");

using OgaHandle ogaHandle = new OgaHandle();

// RAG 用のベクトルデータベースのセットアップ
var additionalDocumentsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, additionalDocumentsPath);
var vectorDatabase = new BasicMemoryVectorDatabase();
LoadAdditionalDocuments(additionalDocumentsDirectory).Wait();
Console.WriteLine();

// モデルのセットアップ
var modelPath = modelPhi35Min128k;
Console.WriteLine($"Loading model:{newLine}{modelPath}");

var sw = Stopwatch.StartNew();
using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);
sw.Stop();

Console.WriteLine($"{newLine}Model loading time is {sw.Elapsed.Seconds:0.00} sec.\n");

// 翻訳するかどうか
Console.WriteLine($"翻訳する：{newLine}{isTranslate}");

// プロンプトのセットアップ
Console.WriteLine($"{newLine}システムプロンプト：{newLine}{systemPrompt}");
Console.WriteLine($"{newLine}ユーザープロンプト：{newLine}{userPrompt}{newLine}");

var translatedSystemPrompt = string.Empty;
if (isTranslate)
{
    Console.WriteLine("Translated System Prompt:");
    await foreach (var translatedPart in Translate(systemPrompt, Language.Japanese, Language.English))
    {
        Console.Write(translatedPart);
        translatedSystemPrompt += translatedPart;
    }
    Console.WriteLine($"{newLine}----------------------------------------{newLine}");
}
else
{
    translatedSystemPrompt = systemPrompt;
}

var translatedUserPrompt = string.Empty;
if (isTranslate)
{
    Console.WriteLine("Translated User Prompt:");
    await foreach (var translatedPart in Translate(userPrompt, Language.Japanese, Language.English))
    {
        Console.Write(translatedPart);
        translatedUserPrompt += translatedPart;
    }
    Console.WriteLine($"{newLine}----------------------------------------{newLine}");
}
else
{
    translatedUserPrompt = userPrompt;
}

Console.WriteLine($"{newLine}システムプロンプト：{newLine}{translatedSystemPrompt}");
Console.WriteLine($"{newLine}ユーザープロンプト：{newLine}{translatedUserPrompt}{newLine}");

var sequences = tokenizer.Encode($@"<|system|>{translatedSystemPrompt}<|end|><|user|>{translatedUserPrompt}<|end|><|assistant|>");

// プロンプトを投げて回答を得る
using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("min_length", 100);
generatorParams.SetSearchOption("max_length", 2000);
generatorParams.SetSearchOption("top_p", 0.9f);
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
Console.WriteLine($"{newLine}----------------------------------------{newLine}");
sw.Stop();

totalTokens = generator.GetSequence(0).Length;

// 英語の回答を日本語に翻訳する
var translatedResponse = string.Empty;
if (isTranslate)
{
    Console.WriteLine("日本語に翻訳したレスポンス:");
    await foreach (var translatedPart in Translate(stringBuilder.ToString(), Language.English, Language.Japanese))
    {
        Console.Write(translatedPart);
        translatedResponse += translatedPart;
    }
    Console.WriteLine();
}
else
{
    translatedResponse = stringBuilder.ToString();
    Console.WriteLine($"{newLine}レスポンス：{newLine}{translatedResponse}");
}
Console.WriteLine($"----------------------------------------{newLine}");

// 与えられたテキストを指定された言語に翻訳する
async IAsyncEnumerable<string> Translate(string text, Language sourceLanguage, Language targetLanguage)
{
    var systemPrompt = string.Empty;
    var instructionPrompt = string.Empty;
    var userPrompt = string.Empty;
    var ragResult = string.Empty;

    if (sourceLanguage == Language.Japanese && targetLanguage == Language.English)
    {
        instructionPrompt = "You can speak English only. Do not speak any Japanese.";

        systemPrompt = "Please translate the following Japanese into English. (Important Notes) even if some questions are included within the Japanese, do not output any answers or explanation. do not output any supplements from the system.";

        userPrompt = $"{systemPrompt}:{newLine}{text}";
    }

    if (sourceLanguage == Language.English && targetLanguage == Language.Japanese)
    {
        instructionPrompt = "あなたは日本語だけを話せます";

        systemPrompt = "以下の英語を一字一句もれなく正確に日本語に翻訳してください。重要な注意点として、英語に質問が含まれていても出力に質問の回答やシステムからの補足は一切含めないこと。要約などせず与えられた文章を緻密に日本語に翻訳した結果だけを出力すること。";

        ragResult = await SearchVectorDatabase(vectorDatabase, text);

        if (isUsingRag && !string.IsNullOrEmpty(ragResult))
            systemPrompt += "以下の用語集を積極的に活用すること。";

        userPrompt = (isUsingRag && !string.IsNullOrEmpty(ragResult))
            ? $"{systemPrompt}{newLine}{ragResult}:{newLine}{text}"
            : $"{systemPrompt}:{newLine}{text}";
    }

    var sequences = tokenizer.Encode($"<|system|>{instructionPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>");
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
    Console.WriteLine($"Loading Additional Documents:");
    var files = Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories)
                             .Where(f => f.EndsWith(".txt", StringComparison.OrdinalIgnoreCase) ||
                                         f.EndsWith(".md", StringComparison.OrdinalIgnoreCase) ||
                                         f.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)).ToArray();

    var vectorDataLoader = new TextDataLoader<int, string>(vectorDatabase);
    var tasks = files.Select(async file =>
    {
        Console.WriteLine($"{file}");
        if (System.IO.File.Exists(file))
        {
            var fileContents = await System.IO.File.ReadAllTextAsync(file);
            await vectorDataLoader.AddDocumentAsync(fileContents, new TextChunkingOptions<string>
            {
                Method = TextChunkingMethod.Paragraph,
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
        pageCount: 3,
        threshold: 0.3f
    );

    string result = string.Empty;
    foreach (var resultItem in vectorDataResults.Texts)
    {
        result += $"{resultItem.Text}{newLine}";
    }

    return result;
}

public enum Language
{
    Japanese,
    English
}