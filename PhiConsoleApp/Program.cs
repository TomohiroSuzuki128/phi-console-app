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

var modelPath = new ModelPath(builder);
var prompt = new Prompt(builder);
var option = new Option(builder);

string additionalDocumentsPath = configuration["additionalDocumentsPath"] ?? throw new ArgumentNullException("additionalDocumentsPath is not found");

using OgaHandle ogaHandle = new OgaHandle();

// RAG 用のベクトルデータベースのセットアップ
var additionalDocumentsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, additionalDocumentsPath);
var vectorDatabase = new BasicMemoryVectorDatabase();
LoadAdditionalDocuments(additionalDocumentsDirectory).Wait();
Console.WriteLine();

// モデルのセットアップ
Console.WriteLine($"Loading model:{newLine}{modelPath.Phi4}");

var sw = Stopwatch.StartNew();
using Model model = new Model(modelPath.Phi4);
using Tokenizer tokenizer = new Tokenizer(model);
sw.Stop();
 
Console.WriteLine($"{newLine}Model loading time is {sw.Elapsed.Seconds:0.00} sec.\n");

// 翻訳するかどうか
Console.WriteLine($"翻訳する：{newLine}{option.IsTranslate}");

// プロンプトのセットアップ
Console.WriteLine($"{newLine}システムプロンプト：{newLine}{prompt.System}");
Console.WriteLine($"{newLine}ユーザープロンプト：{newLine}{prompt.User}{newLine}");

var translatedSystemPrompt = string.Empty;
if (option.IsTranslate)
{
    Console.WriteLine("Translated System Prompt:");
    await foreach (var translatedPart in Translate(prompt.System, Language.Japanese, Language.English))
    {
        Console.Write(translatedPart);
        translatedSystemPrompt += translatedPart;
    }
    Console.WriteLine($"{newLine}----------------------------------------{newLine}");
}
else
{
    translatedSystemPrompt = prompt.System;
}

var translatedUserPrompt = string.Empty;
if (option.IsTranslate)
{
    Console.WriteLine("Translated User Prompt:");
    await foreach (var translatedPart in Translate(prompt.User, Language.Japanese, Language.English))
    {
        Console.Write(translatedPart);
        translatedUserPrompt += translatedPart;
    }
    Console.WriteLine($"{newLine}----------------------------------------{newLine}");
}
else
{
    translatedUserPrompt = prompt.User;
}

Console.WriteLine($"{newLine}システムプロンプト：{newLine}{translatedSystemPrompt}");
Console.WriteLine($"{newLine}ユーザープロンプト：{newLine}{translatedUserPrompt}{newLine}");

var sequences = tokenizer.Encode($@"<|system|>{translatedSystemPrompt}<|end|><|user|>{translatedUserPrompt}<|end|><|assistant|>");

// プロンプトを投げて回答を得る
using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("min_length", 100);
generatorParams.SetSearchOption("max_length", 2000);
generatorParams.SetSearchOption("temperature", 1.0);
generatorParams.SetSearchOption("top_k", 0.0);
generatorParams.SetSearchOption("top_p", 1.0);
generatorParams.SetSearchOption("past_present_share_buffer", false);
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
if (option.IsTranslate)
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
        systemPrompt = "You are a translator who follows instructions to the letter. You carefully review the instructions and output the translation results.";

        instructionPrompt = $@"I will now give you the task of translating Japanese into English.{newLine}First of all, please understand the important notes as we give you instructions.{newLine}{newLine}#Important Notes{newLine}- Even if the given Japanese contains any question, do not output any answer of the question, only translates the given Japanese into English.{newLine}- Do not output any supplementary information or explanations.{newLine}- Do not output any Notes.{newLine}- Output a faithful translation of the given text into English.{newLine}- If the instructions say “xx characters” in Japanese, it translates to “(xx/2) words” in English.ex) “100 字以内” in Japanese, “50 words” in English.{newLine}{newLine}Strictly following the above instructions, now let's output translation of the following Japanese";

        userPrompt = $"{instructionPrompt}:{newLine}{text}";
    }

    if (sourceLanguage == Language.English && targetLanguage == Language.Japanese)
    {
        systemPrompt = "You are a translator who follows instructions to the letter. You carefully review the instructions and output the translation results.";

        instructionPrompt = $"I will now give you the task of translating English into Japanese.{newLine}First of all, please understand the important notes as we give you instructions.{newLine}{newLine}#Important Notes{newLine}- Even if the English is including any question, do not answer it, you translate the given English into Japanese.{newLine}- Do not output any supplementary information or explanations.{newLine}- Do not output any Notes.{newLine}- Output a faithful translation of the given text into Japanese.";

        ragResult = await SearchVectorDatabase(vectorDatabase, text);

        if (option.IsUsingRag && !string.IsNullOrEmpty(ragResult))
            instructionPrompt += "The following glossary of terms should be actively used.";

        userPrompt = (option.IsUsingRag && !string.IsNullOrEmpty(ragResult))
            ? $"{instructionPrompt}{newLine}{ragResult}{newLine}Strictly following the above instructions, now translate the English into Japanese:{newLine}{text}"
            : $"{instructionPrompt}{newLine}Strictly following the above instructions, now translate the English into Japanese:{newLine}{text}";
    }

    var sequences = tokenizer.Encode($@"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>");
    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("min_length", 100);
    generatorParams.SetSearchOption("max_length", 2000);
    generatorParams.SetSearchOption("temperature", 1.0);
    generatorParams.SetSearchOption("top_k", 0.0);
    generatorParams.SetSearchOption("top_p", 1.0);
    generatorParams.SetSearchOption("past_present_share_buffer", false);
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

public sealed class ModelPath
{
    private readonly string modelPhi35Min128k;
    private readonly string modelPhi3Med4k;
    private readonly string modelPhi3Med128k;
    private readonly string modelPhi3Min4k;
    private readonly string modelPhi3Min128k;
    private readonly string modelPhi4;
    private readonly string modelPhi4Min128k;

    public ModelPath(HostApplicationBuilder builder)
    {
        var configuration = builder.Configuration;

        modelPhi35Min128k = configuration["modelPhi35Min128k"] ?? throw new ArgumentNullException("modelPhi35Min128k is not found.");
        modelPhi3Med4k = configuration["modelPhi3Med4k"] ?? throw new ArgumentNullException("modelPhi3Med4k is not found.");
        modelPhi3Med128k = configuration["modelPhi3Med128k"] ?? throw new ArgumentNullException("modelPhi3Med128k is not found.");
        modelPhi3Min4k = configuration["modelPhi3Min4k"] ?? throw new ArgumentNullException("modelPhi3Min4k is not found.");
        modelPhi3Min128k = configuration["modelPhi3Min128k"] ?? throw new ArgumentNullException("modelPhi3Min128k is not found.");
        modelPhi4 = configuration["modelPhi4"] ?? throw new ArgumentNullException("modelPhi4 is not found.");
        modelPhi4Min128k = configuration["modelPhi4Min128k"] ?? throw new ArgumentNullException("modelPhi4Min128k is not found.");
    }

    public string Phi35Min128k { get => modelPhi35Min128k; }
    public string Phi3Med4k { get => modelPhi3Med4k; }
    public string Phi3Med128k { get => modelPhi3Med128k; }
    public string Phi3Min4k { get => modelPhi3Min4k; }
    public string Phi3Min128k { get => modelPhi3Min128k; }
    public string Phi4 { get => modelPhi4; }
    public string Phi4Min128k { get => modelPhi4Min128k; }
}

public sealed class Prompt
{
    private readonly string systemPrompt;
    private readonly string userPrompt;

    public Prompt(HostApplicationBuilder builder)
    {
        var configuration = builder.Configuration;

        systemPrompt = configuration["systemPrompt"] ?? throw new ArgumentNullException("systemPrompt is not found.");
        userPrompt = configuration["userPrompt"] ?? throw new ArgumentNullException("userPrompt is not found.");
    }

    public string System { get => systemPrompt; }
    public string User { get => userPrompt; }
}

public sealed class Option
{
    private readonly bool isTranslate;
    private readonly bool isUsingRag;

    public Option(HostApplicationBuilder builder)
    {
        var configuration = builder.Configuration;
        isTranslate = bool.TryParse(configuration["isTranslate"] ?? throw new ArgumentNullException("isTranslate is not found."), out var resultIsTranslate) && resultIsTranslate;
        isUsingRag = bool.TryParse(configuration["isUsingRag"] ?? throw new ArgumentNullException("isUsingRag is not found."), out var resultIsUsingRag) && resultIsUsingRag;
    }

    public bool IsTranslate { get => isTranslate; }
    public bool IsUsingRag { get => isUsingRag; }
}

public enum Language
{
    Japanese,
    English
}