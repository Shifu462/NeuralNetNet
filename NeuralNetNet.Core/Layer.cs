using System;
using System.Collections;
using System.Linq;
using NeuralNetNet.ActivationFunctions;

namespace NeuralNetNet
{
    /// <summary>
    /// Abstract model of a layer.
    /// </summary>
    public class Layer : IEnumerable
    {
        /// <summary>
        /// Neurons inside of this layer.
        /// </summary>
        protected Neuron[] Neurons { get; set; }

        /// <summary>
        /// Random number generator used for filling SynapseWeights with random values.
        /// </summary>
        protected static Random rnd = new Random();

        /// <summary>
        /// Count of neurons in current layer.
        /// </summary>
        public long Count => Neurons.Length;

        /// <summary>
        /// Simple indexator. Shortcut for Neuron[] Neurons.
        /// </summary>
        public Neuron this[int index]
        {
            get => Neurons[index];
        }

        /// <summary>
        /// So you can foreach Layer neurons.
        /// </summary>
        public IEnumerator GetEnumerator() => this.Neurons.GetEnumerator();

        /// <summary>
        /// Pointer to the previous layer in the network.
        /// </summary>
        public Layer PreviousLayer { get; internal set; }

        /// <summary>
        /// Pointer to the next layer in the network.
        /// </summary>
        public Layer NextLayer { get; internal set; }

        /// <summary>
        /// Activation function that this layer uses.
        /// </summary>
        public IActivation Activation { get; protected set; } = new Sigmoid();

        /// <summary>
        /// Fill neurons on this layer with calculated values.
        /// </summary>
        public virtual void Process()
        {
            for (int i = 0; i < this.Count; i++)
            {
                double sum = 0;

                for (int j = 0; j < PreviousLayer.Count; j++)
                    sum += this[i].SynapsesWeights[j] * PreviousLayer[j].Value;

                Neurons[i].Value = Activation.Activate(sum);
            }
        }

        public double[] GetOutput() => NextLayer == null
                                     ? Neurons.Select(n => n.Value).ToArray()
                                     : throw new NullReferenceException("Can't get output from hidden or input layer.");

        public void Train(double learningRate = 1, double moment = 1)
        {
            // set neurons delta
            for (int hi = 0; hi < this.Count; hi++)
            {
                // sum of | delta * w
                double sum = 0;
                foreach (Neuron nextNeuron in this.NextLayer)
                {
                    for (int w = 0; w < nextNeuron.SynapsesWeights.Length; w++)
                    {
                        if (hi == w) // find synapse that connects hiddenNeuron with nextNeuron
                            sum += nextNeuron.Delta * nextNeuron.SynapsesWeights[w];
                    }
                }

                Neuron hiddenNeuron = this[hi];
                hiddenNeuron.Delta = sum * Activation.Activate(hiddenNeuron.Value, derivative: true);

                // change weights
                for (int sw = 0; sw < hiddenNeuron.SynapsesWeights.Length; sw++)
                {
                    double prevNeuronValue = this.PreviousLayer[sw].Value;
                    hiddenNeuron.SynapsesWeights[sw] += prevNeuronValue * hiddenNeuron.Delta * learningRate;
                }
            }
        }

        public Layer(int neuronsCount, IActivation activation)
        {
            Neurons = new Neuron[neuronsCount];
            Activation = activation;
        }

        internal void FillLayer()
        {
            // Check if no synapses behind this layer (input layer check)
            long synapseCount = PreviousLayer != null ? PreviousLayer.Count : 0;

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neuron currentNeuron = Neurons[i] = new Neuron(synapseCount);

                if (PreviousLayer == null)
                    continue;

                for (int w = 0; w < currentNeuron.SynapsesWeights.Length; w++)
                    currentNeuron.SynapsesWeights[w] = rnd.NextDouble();

            }
        }
    }
}