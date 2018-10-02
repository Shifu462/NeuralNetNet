using System.Collections;
using System.Collections.Generic;

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
        public Neuron[] Neurons { get; protected set; }

        public Neuron this[int index]
        {
            get => Neurons[index];
        }

        public long Count => Neurons.Length;

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

                this[i].SetValueWithSigmoid(sum);
            }
        }

        public IEnumerator GetEnumerator() => this.Neurons.GetEnumerator();

        /// <summary>
        /// Pointer to the previous layer in the network.
        /// </summary>
        public Layer PreviousLayer { get; protected set; }
        
        /// <summary>
        /// Pointer to the next layer in the network.
        /// </summary>
        public Layer NextLayer { get; set; }

        public Layer(int neuronsCount, Layer previous)
        {
            Neurons = new Neuron[neuronsCount];

            PreviousLayer = previous;

            // Check if no synapses behind this layer (input layer check)
            long synapseCount = PreviousLayer != null ? PreviousLayer.Count : 0;

            for (int i = 0; i < neuronsCount; i++)
                Neurons[i] = new Neuron(synapseCount);
        }


    }
}