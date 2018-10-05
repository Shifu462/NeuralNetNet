using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetNet.NeuralNetwork
{
    public class Perceptron : NeuralNetwork
    {
        public Perceptron(Layer inputLayer, IEnumerable<Layer> hiddenLayers, Layer outputLayer)
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

        public override void Train(List<TrainSet> trainSetList, int maxEpoch
            , double learningRate = 1
            , double moment = 1)
        {
            double[] idealOutputs = trainSetList.Select(t => t.Output).ToArray();

            for (int ep = 0; ep < maxEpoch; ep++)
            {
                for (int ts = 0; ts < trainSetList.Count; ts++)
                {
                    TrainSet currentSet = trainSetList[ts];
                    double actual = this.Predict(currentSet.Input)[0];
                    double error = currentSet.Output - actual; // ideal - actual

                    foreach (Neuron outn in OutputLayer)
                        outn.Delta = error * OutputLayer.Activation.Activate(outn.Value, derivative: true);

                    // train each hidden layer
                    Layer backpropLayer = OutputLayer;
                    while ((backpropLayer = backpropLayer.PreviousLayer) != null)
                        backpropLayer.Train(learningRate, moment);

                    // Change OutputLayer weights.
                    foreach (Neuron outNeuron in OutputLayer)
                    {
                        for (int s = 0; s < outNeuron.SynapsesWeights.Length; s++)
                        {
                            double prevNeuronValue = OutputLayer.PreviousLayer[s].Value;
                            outNeuron.SynapsesWeights[s] += prevNeuronValue * outNeuron.Delta * learningRate;
                        }
                    }

                    if (ts == 0 && ep % 1000 == 0) Console.WriteLine($"Ep #{ep} | {Math.Abs(error)}");
                }


            }

        }

        public override double[] Predict(params double[] inputs)
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
