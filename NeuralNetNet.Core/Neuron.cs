using System;

namespace NeuralNetNet
{
    public class Neuron
    {
        public double Value { get; set; }

        public double Delta { get; set; }

        public void SetValueWithSigmoid(double value)
        {
            this.Value = MathFunctions.ApproxSigmoid(value);
        }

        public double[] SynapsesWeights { get; set; }

        public Neuron(long synapseCount)
        {
            Random rnd = new Random();

            SynapsesWeights = new double[synapseCount];

            for (int i = 0; i < synapseCount; i++)
                SynapsesWeights[i] = rnd.NextDouble();
                // SynapsesWeights[i] = 0;
        }
    }
}
