using System.Collections.Generic;

namespace NeuralNetNet.NeuralNetwork
{
    public abstract class NeuralNetwork
    {
        public Layer InputLayer { get; protected set; }

        public Layer OutputLayer { get; protected set; }

        public abstract void Train(List<TrainSet> trainSetList, int maxEpoch, double learningRate = 1, double moment = 1);

        public abstract double[] Predict(params double[] inputs);

    }
}