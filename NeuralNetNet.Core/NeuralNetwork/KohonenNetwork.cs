using System;
using System.Linq;

namespace NeuralNetNet.NeuralNetwork
{
    public class KohonenNetwork
    {
        public int InputsCount { get; protected set; }

        /// <summary>
        /// Kohonen layer.
        /// </summary>
        public readonly Neuron[] Neurons;

        public KohonenNetwork(int inputsCount, int answersCount)
        {
            InputsCount = inputsCount;
            Neurons = new Neuron[answersCount];

            Random rnd = new Random();
            for (int n = 0; n < Neurons.Length; n++)
            {
                Neurons[n] = new Neuron(inputsCount);

                Neuron neuron = Neurons[n];
                for (int w = 0; w < neuron.IncomingWeights.Length; w++)
                    neuron.IncomingWeights[w] = rnd.NextDouble();
            }
                
                
        }

        /// <summary>
        /// Returns index of the result output neuron.
        /// </summary>
        public int Handle(params int[] inputs)
        {
            if (inputs.Length != InputsCount)
                throw new ArgumentException();

            // Winner-take-all.
            for (int i = 0; i < InputsCount; i++)
                foreach (Neuron currentNeuron in Neurons)
                    currentNeuron.Value += currentNeuron.IncomingWeights[i] * inputs[i];

            // Get index of the result output neuron.
            var maxIndex = 0;
            for (var i = 1; i < Neurons.Length; i++)
            {
                if (Neurons[i].Value > Neurons[maxIndex].Value)
                    maxIndex = i;
            }

            // Clear values.
            foreach (Neuron n in Neurons)
                n.Value = 0;

            return maxIndex;
        }

        /// <summary>
        /// Trains on a single set.
        /// Returns error: true if no error, false if error.
        /// </summary>
        public bool Study(int[] inputs, int correctAnswer, double learningRate = 0.5)
        {
            Neuron neuron = Neurons[correctAnswer];

            for (int i = 0; i < neuron.IncomingWeights.Length; i++)
                neuron.IncomingWeights[i] += learningRate * (inputs[i] - neuron.IncomingWeights[i]);

            return Handle(inputs) == correctAnswer;
        }
    }
}
