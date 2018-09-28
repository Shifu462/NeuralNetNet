using System;

namespace NeuralNetNet
{
    public class Neuron
    {
        protected double _value;

        public double Value
        {
            get => _value;
            set => _value = value;
        }

        public double Delta { get; set; }

        public void SetValueWithSigmoid(double value)
        {
            this._value = MathFunctions.ApproxSigmoid(value);
        }

        public double[] SynapsesWeights { get; set; }

        public Neuron(long synapseCount)
        {
            Random rnd = new Random();

            SynapsesWeights = new double[synapseCount];

            for (int i = 0; i < synapseCount; i++)
                SynapsesWeights[i] = rnd.NextDouble();
        }

        
    }
}
