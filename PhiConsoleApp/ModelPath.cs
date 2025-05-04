using Microsoft.Extensions.Configuration;

public sealed class ModelPath
{
    private readonly string modelPhi35Min128k;
    private readonly string modelPhi3Med4k;
    private readonly string modelPhi3Med128k;
    private readonly string modelPhi3Min4k;
    private readonly string modelPhi3Min128k;
    private readonly string modelPhi4;
    private readonly string modelPhi4Gpu;
    private readonly string modelPhi4Min128k;
    private readonly string modelPhi4Min128kGpu;

    public ModelPath(ConfigurationManager configuration)
    {
        modelPhi35Min128k = configuration["modelPhi35Min128k"] ?? throw new ArgumentNullException("modelPhi35Min128k is not found.");
        modelPhi3Med4k = configuration["modelPhi3Med4k"] ?? throw new ArgumentNullException("modelPhi3Med4k is not found.");
        modelPhi3Med128k = configuration["modelPhi3Med128k"] ?? throw new ArgumentNullException("modelPhi3Med128k is not found.");
        modelPhi3Min4k = configuration["modelPhi3Min4k"] ?? throw new ArgumentNullException("modelPhi3Min4k is not found.");
        modelPhi3Min128k = configuration["modelPhi3Min128k"] ?? throw new ArgumentNullException("modelPhi3Min128k is not found.");
        modelPhi4 = configuration["modelPhi4"] ?? throw new ArgumentNullException("modelPhi4 is not found.");
        modelPhi4Gpu = configuration["modelPhi4Gpu"] ?? throw new ArgumentNullException("modelPhi4Gpu is not found.");
        modelPhi4Min128k = configuration["modelPhi4Min128k"] ?? throw new ArgumentNullException("modelPhi4Min128k is not found.");
        modelPhi4Min128kGpu = configuration["modelPhi4Min128kGpu"] ?? throw new ArgumentNullException("modelPhi4Min128kGpu is not found.");
    }

    public string Phi35Min128k { get => modelPhi35Min128k; }
    public string Phi3Med4k { get => modelPhi3Med4k; }
    public string Phi3Med128k { get => modelPhi3Med128k; }
    public string Phi3Min4k { get => modelPhi3Min4k; }
    public string Phi3Min128k { get => modelPhi3Min128k; }
    public string Phi4 { get => modelPhi4; }
    public string Phi4Gpu { get => modelPhi4Gpu; }
    public string Phi4Min128k { get => modelPhi4Min128k; }
    public string Phi4Min128kGpu { get => modelPhi4Min128kGpu; }
}
