using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.ML.OnnxRuntimeGenAI;

var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.Sources.Clear();
builder.Configuration
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
    .AddJsonFile($"appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)
    .Build();

var configuration = builder.Configuration;
var modelPath = new ModelPath(configuration);
var prompt = new Prompt(configuration);
var option = new Option(configuration);
var additionalDocumentsPath = configuration["additionalDocumentsPath"] ?? throw new ArgumentNullException("additionalDocumentsPath is not found");

using (var ogaHandle = new OgaHandle())
using (var model = new Model(modelPath.Phi4))
{
    TextGenerator textGenerator = new(args, model, prompt, option, additionalDocumentsPath);
    await textGenerator.Generate();
}