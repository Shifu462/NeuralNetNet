using System.Collections.Generic;

namespace NeuralNetNet.NeuralNetwork
{
    public class KohonenNetwork : NeuralNetwork
    {
        public int InputsCount { get; protected set; }

        public readonly Neuron[] Neurons;

        public override void Train(List<TrainSet> trainSet, int maxEpoch, double learningRate = 1, double moment = 1)
        {
            
        }

        /// <summary>
        /// Returns single result. Result array will contain only one element.
        /// </summary>
        public override double[] Handle(params double[] inputs)
        {

            for (int i = 0; i < InputsCount; i++)
            {
                foreach (Neuron currentNeuron in Neurons)
                {
                    currentNeuron.Value = currentNeuron.IncomingWeights[i] * inputs[i];


                }
            }


            foreach (Neuron n in Neurons)
                n.Value = 0;

            return new double[0];
        }

        protected override double Train(TrainSet trainSet, double learningRate = 1, double moment = 1)
        {
            return -1;
        }

        public KohonenNetwork(int inputsCount, int neuronsCount)
        {
            InputsCount = inputsCount;
            Neurons = new BackpropNeuron[neuronsCount];
        }
    }
}
