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
        protected readonly BackpropNeuron[] Neurons;

        /// <summary>
        /// Simple indexator. Shortcut for Neuron[] Neurons.
        /// </summary>
        public BackpropNeuron this[int index]
        {
            get => Neurons[index];
        }

        /// <summary>
        /// Count of neurons in current layer.
        /// </summary>
        public long Count => Neurons.Length;

        /// <summary>
        /// Random number generator used for filling SynapseWeights with random values.
        /// </summary>
        protected static Random rnd = new Random();
        
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
                    sum += this[i].IncomingWeights[j] * PreviousLayer[j].Value;

                Neurons[i].Value = Activation.Activate(sum);
            }
        }

        /// <summary>
        /// If it's an output layer, returns result.
        /// </summary>
        public double[] GetOutput() => NextLayer == null
                                     ? Neurons.Select(n => n.Value).ToArray()
                                     : throw new ArgumentException("Can't get output from hidden or input layer.");

        public void Train(double learningRate = 1, double moment = 1)
        {
            // set neurons delta
            for (int hi = 0; hi < this.Count; hi++)
            {
                // sum of | delta * w
                double sum = 0;
                foreach (BackpropNeuron nextNeuron in this.NextLayer)
                {
                    for (int w = 0; w < nextNeuron.IncomingWeights.Length; w++)
                    {
                        if (hi == w) // find synapse that connects hiddenNeuron with nextNeuron
                            sum += nextNeuron.Delta * nextNeuron.IncomingWeights[w];
                    }
                }

                BackpropNeuron hiddenNeuron = this[hi];
                hiddenNeuron.Delta = sum * Activation.Activate(hiddenNeuron.Value, derivative: true);

                // change weights
                for (int sw = 0; sw < hiddenNeuron.IncomingWeights.Length; sw++)
                {
                    double prevNeuronValue = this.PreviousLayer[sw].Value;
                    hiddenNeuron.IncomingWeights[sw] += prevNeuronValue * hiddenNeuron.Delta * learningRate;
                }
            }
        }

        public Layer(int neuronsCount, IActivation activation)
        {
            Neurons = new BackpropNeuron[neuronsCount];
            Activation = activation;
        }

        internal void FillLayer()
        {
            // Check if no synapses behind this layer (input layer check)
            long synapseCount = PreviousLayer != null ? PreviousLayer.Count : 0;

            for (int i = 0; i < Neurons.Length; i++)
            {
                BackpropNeuron currentNeuron = Neurons[i] = new BackpropNeuron(synapseCount);

                if (PreviousLayer == null)
                    continue;

                for (int w = 0; w < currentNeuron.IncomingWeights.Length; w++)
                    currentNeuron.IncomingWeights[w] = rnd.NextDouble();

            }
        }
    }
}