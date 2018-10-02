using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetNet
{
    public class Perceptron
    {
        public Layer InputLayer { get; protected set; }

        public Layer OutputLayer { get; protected set; }

        public Perceptron(int inputSize, int outputSize, int hiddenCount = 0, int hiddenSize = 0)
        {
            InputLayer = new Layer(inputSize, null);

            Layer currentLayer = InputLayer;
            for (int i = 0; i < hiddenCount; i++)
            {
                currentLayer.NextLayer = new Layer(hiddenSize, previous: currentLayer);
                currentLayer = currentLayer.NextLayer;
            }

            OutputLayer = new Layer(outputSize, previous: currentLayer);
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
                    double actual = this.Predict(currentSet.Input)[0];

                    double error = currentSet.Output - actual; // ideal - actual

                    foreach (Neuron outn in OutputLayer)
                    {
                        // delta for output
                        outn.Delta = error * (outn.Value * (1 - outn.Value)); // value is already sigmoid(x).
                    }

                    // foreach Hidden Layer
                    Layer backpropLayer = OutputLayer.PreviousLayer;
                    while (backpropLayer != null)
                    {
                        // set neurons delta
                        for (int hi = 0; hi < backpropLayer.Count; hi++)
                        {
                            Neuron hiddenNeuron = backpropLayer[hi];

                            // sum of | delta * w
                            double sum = 0;
                            foreach (Neuron nextNeuron in backpropLayer.NextLayer)
                            {
                                for (int w = 0; w < nextNeuron.SynapsesWeights.Length; w++)
                                {
                                    if (hi == w) // find synapse that connects hiddenNeuron with nextNeuron
                                        sum += nextNeuron.Delta * nextNeuron.SynapsesWeights[w];
                                }

                            }

                            hiddenNeuron.Delta = sum * (hiddenNeuron.Value * (1 - hiddenNeuron.Value));

                            // change weights
                            for (int sw = 0; sw < hiddenNeuron.SynapsesWeights.Length; sw++)
                            {
                                double prevNeuronValue = backpropLayer.PreviousLayer[sw].Value;
                                hiddenNeuron.SynapsesWeights[sw] += prevNeuronValue * hiddenNeuron.Delta * learningRate;
                            }
                        }

                        // go back
                        backpropLayer = backpropLayer.PreviousLayer;
                    }

                    // Change OutputLayer weights.
                    foreach (Neuron outNeuron in OutputLayer)
                    {
                        for (int s = 0; s < outNeuron.SynapsesWeights.Length; s++)
                        {
                            double prevNeuronValue = OutputLayer.PreviousLayer[s].Value;
                            outNeuron.SynapsesWeights[s] += prevNeuronValue * outNeuron.Delta * learningRate;
                        }
                    }

                    if (ts % 50000 == 0) Console.WriteLine($"Ep #{ep} | {Math.Abs(error)}");
                }


            }

        }

        public double[] Predict(params double[] inputs)
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

            return OutputLayer.Neurons.Select(n => n.Value).ToArray();
        }
    }
}
