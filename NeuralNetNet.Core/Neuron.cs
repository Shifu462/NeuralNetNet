using System;

namespace NeuralNetNet
{
    public class Neuron
    {
        public double Value { get; set; }

        public double Delta { get; set; }

        public double[] SynapsesWeights { get; set; }

        public Neuron(long synapseCount)
        {
            SynapsesWeights = new double[synapseCount];
        }
    }
}
