using System;
using System.Collections.Generic;

namespace NeuralNetNet.NeuralNetwork
{
    public class Convolutional : NeuralNetwork
    {
        public override double[] Predict(params double[] inputs) => throw new NotImplementedException();
        public override void Train(List<TrainSet> trainSetList, int maxEpoch, double learningRate = 1, double moment = 1) => throw new NotImplementedException();
    }
}
