using System.Collections.Generic;

namespace NeuralNetNet.NeuralNetwork
{
    public abstract class NeuralNetwork
    {
        public Layer InputLayer { get; protected set; }

        public Layer OutputLayer { get; protected set; }

        /// <summary>
        /// Train on a list of trainsets.
        /// </summary>
        /// <param name="trainSetList">List of trainsets used to train network.</param>
        /// <param name="maxEpoch">Count of iterations.</param>
        /// <param name="learningRate">Learn </param>
        /// <param name="moment"></param>
        public abstract void Train(List<TrainSet> trainSetList, int maxEpoch, double learningRate = 1, double moment = 1);

        /// <summary>
        /// Trains on a single trainset.
        /// </summary>
        protected abstract double Train(TrainSet trainSet, double learningRate = 1, double moment = 1);

        /// <summary>
        /// Uses input to get result.
        /// </summary>
        public abstract double[] Handle(params double[] inputs);

    }
}