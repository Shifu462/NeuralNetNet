namespace NeuralNetNet.ActivationFunctions
{
    public interface IActivation
    {
        double Activate(double value, bool derivative = false);
    }
}