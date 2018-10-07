using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetNet.NeuralNetwork
{
    public class FeedForward
    {
        public Layer InputLayer { get; protected set; }

        public Layer OutputLayer { get; protected set; }

        public FeedForward(Layer inputLayer, IEnumerable<Layer> hiddenLayers, Layer outputLayer)
        {
            InputLayer = inputLayer;
            InputLayer.FillLayer();

            Layer currentLayer = InputLayer;
            foreach (Layer layer in hiddenLayers)
            {
                layer.PreviousLayer = currentLayer;
                layer.FillLayer();
                currentLayer.NextLayer = layer;

                if (currentLayer.NextLayer != null)
                    currentLayer = currentLayer.NextLayer;
            }

            outputLayer.PreviousLayer = hiddenLayers.Last();
            outputLayer.FillLayer();
            OutputLayer = outputLayer;

            currentLayer.NextLayer = OutputLayer;
        }
        
        public void Train(List<TrainSet> trainSetList, int maxEpoch
            , double learningRate = 1
            , double moment = 1)
        {
            double[] idealOutputs = trainSetList.Select(t => t.Output).ToArray();

            for (int ep = 0; ep < maxEpoch; ep++)
            {
                for (int ts = 0; ts < trainSetList.Count; ts++)
                {
                    TrainSet currentSet = trainSetList[ts];

                    double error = Study(currentSet, learningRate, moment);

                    if (ts == 0 && ep % 1000 == 0) Console.WriteLine($"Ep #{ep} | {Math.Abs(error)}");
                }
            }
        }
        
        /// <summary>
        /// Trains on a single set.
        /// </summary>
        public double Study(TrainSet currentSet, double learningRate = 1, double moment = 1)
        {

            double actual = this.Handle(currentSet.Input)[0];
            double error = currentSet.Output - actual; // ideal - actual

            foreach (BackpropNeuron outn in OutputLayer)
                outn.Delta = error * OutputLayer.Activation.Activate(outn.Value, derivative: true);

            // train each hidden layer
            Layer backpropLayer = OutputLayer;
            while ((backpropLayer = backpropLayer.PreviousLayer) != null)
                backpropLayer.Train(learningRate, moment);

            // Change OutputLayer weights.
            foreach (BackpropNeuron outNeuron in OutputLayer)
            {
                for (int s = 0; s < outNeuron.IncomingWeights.Length; s++)
                {
                    double prevNeuronValue = OutputLayer.PreviousLayer[s].Value;
                    outNeuron.IncomingWeights[s] += prevNeuronValue * outNeuron.Delta * learningRate;
                }
            }
            
            return error;
        }
        
        public double[] Handle(params double[] inputs)
        {
            if (inputs.Length != InputLayer.Count)
                throw new ArgumentException();

            for (int i = 0; i < inputs.Length; i++)
                InputLayer[i].Value = MathFunctions.Normalize(inputs[i]);

            Layer currentWorkingLayer = InputLayer.NextLayer;
            do
            {
                currentWorkingLayer.Process();
                currentWorkingLayer = currentWorkingLayer.NextLayer; // move on
            }
            while (currentWorkingLayer != null); // if null then it's an output layer.

            return OutputLayer.GetOutput();
        }
    }
}
