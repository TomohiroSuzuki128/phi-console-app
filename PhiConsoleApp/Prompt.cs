using Microsoft.Extensions.Configuration;

public sealed class Prompt
{
    private readonly string systemPrompt;
    private readonly string userPrompt;

    public Prompt(ConfigurationManager configuration)
    {
        systemPrompt = configuration["systemPrompt"] ?? throw new ArgumentNullException("systemPrompt is not found.");
        userPrompt = configuration["userPrompt"] ?? throw new ArgumentNullException("userPrompt is not found.");
    }

    public string System { get => systemPrompt; }
    public string User { get => userPrompt; }
}

