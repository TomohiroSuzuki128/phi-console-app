using Microsoft.Extensions.Configuration;

public sealed class Option
{
    private readonly bool isTranslate;
    private readonly bool isUsingRag;

    public Option(ConfigurationManager configuration)
    {
        isTranslate = bool.TryParse(configuration["isTranslate"] ?? throw new ArgumentNullException("isTranslate is not found."), out var resultIsTranslate) && resultIsTranslate;
        isUsingRag = bool.TryParse(configuration["isUsingRag"] ?? throw new ArgumentNullException("isUsingRag is not found."), out var resultIsUsingRag) && resultIsUsingRag;
    }

    public bool IsTranslate { get => isTranslate; }
    public bool IsUsingRag { get => isUsingRag; }
}


