using System.Collections;
using System.Collections.Generic;

namespace NeuralNetNet.Layers
{
    /// <summary>
    /// Abstract model of a layer.
    /// </summary>
    public abstract class Layer : IEnumerable
    {
        /// <summary>
        /// Neurons inside of this layer.
        /// </summary>
        public List<Neuron> Neurons { get; set; } = new List<Neuron>();

        public Neuron this[int index]
        {
            get => Neurons[index];
        }

        public long Count => Neurons.Count;

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

        public IEnumerator GetEnumerator() => ((IEnumerable)this.Neurons).GetEnumerator();

        /// <summary>
        /// Pointer to a previous layer in the network.
        /// </summary>
        public Layer PreviousLayer { get; protected set; }

        public Layer(int neuronsCount, Layer previous)
        {
            PreviousLayer = previous;

            // Check if no synapses behind this layer (input layer check)
            long synapseCount = PreviousLayer != null ? PreviousLayer.Count : 0;

            for (int i = 0; i < neuronsCount; i++)
                Neurons.Add(new Neuron(synapseCount));
        }


    }
}