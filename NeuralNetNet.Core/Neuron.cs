
namespace NeuralNetNet
{
    /// <summary>
    /// Clean neuron object.
    /// </summary>
    public class Neuron
    {
        public double Value { get; set; }

        public double[] IncomingWeights { get; set; }

        public Neuron(long synapseCount)
        {
            IncomingWeights = new double[synapseCount];
        }
    }
}