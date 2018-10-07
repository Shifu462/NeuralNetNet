
namespace NeuralNetNet
{
    /// <summary>
    /// Neuron object containing delta field. 
    /// Delta is used in backpropagation training algorithms.
    /// </summary>
    public class BackpropNeuron : Neuron
    {
        public double Delta { get; set; }

        public BackpropNeuron(long synapseCount) : base(synapseCount)
        {
            
        }
    }
}
