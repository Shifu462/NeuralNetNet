using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetNet.Layers;

namespace NeuralNetNet
{
    public class NeuralNetwork
    {
        public InputLayer InputLayer { get; protected set; }

        public HiddenLayer HiddenLayer { get; protected set; }

        public OutputLayer OutputLayer { get; protected set; }

        public NeuralNetwork(int inputSize, int outputSize, int hiddenCount = 0, int hiddenSize = 0)
        {
            InputLayer = new InputLayer(inputSize);

            if (hiddenCount == 0)
            {
                OutputLayer = new OutputLayer(outputSize, InputLayer);
                return;
            }

            HiddenLayer = new HiddenLayer(hiddenSize, InputLayer);

            OutputLayer = new OutputLayer(outputSize, HiddenLayer);
            HiddenLayer.NextLayer = OutputLayer;
        }

        public void Train(List<XorTrainSet> trainSetList, int maxEpoch
            , double learningRate = 10
            , double moment = 1)
        {
            double[] idealOutputs = trainSetList.Select(t => t.Output).ToArray();

            for (int ep = 0; ep < maxEpoch; ep++)
            {
                for (int ts = 0; ts < trainSetList.Count; ts++)
                {
                    var currentSet = trainSetList[ts];
                    double actual = this.Predict(currentSet.Input)[0];

                    double error = currentSet.Output - actual; // ideal - actual
                    
                    foreach (Neuron outn in OutputLayer)
                    {
                        // delta for output
                        outn.Delta = error * (outn.Value * (1 - outn.Value));
                    }

                    // foreach Hidden Layer
                    // set neurons delta
                    for (int hi = 0; hi < HiddenLayer.Count; hi++)
                    {
                        Neuron hiddenNeuron = HiddenLayer[hi];

                        double sum = 0;

                        // w1 * d1 + ..

                        // Count sum of 
                        foreach (Neuron nextNeuron in HiddenLayer.NextLayer)
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
                            double prevNeuronValue = HiddenLayer.PreviousLayer[sw].Value;
                            hiddenNeuron.SynapsesWeights[sw] += prevNeuronValue * hiddenNeuron.Delta * learningRate;
                        }


                    }


                    //foreach outNeuron in outputlayer

                    // change weights
                    Neuron outNeuron = OutputLayer[0];
                    for (int s = 0; s < outNeuron.SynapsesWeights.Length; s++)
                    {
                        double prevNeuronValue = OutputLayer.PreviousLayer[s].Value;
                        double grad = prevNeuronValue * outNeuron.Delta;

                        double diff = grad * learningRate; // + moment * 

                        outNeuron.SynapsesWeights[s] += diff; 
                    }

                    this.ClearDeltas();
                }


            }

        }

        public double[] Predict(params double[] inputs)
        {
            if (inputs.Length != InputLayer.Count)
                throw new ArgumentException();

            for (int i = 0; i < inputs.Length; i++)
                InputLayer[i].Value = inputs[i];

            InputLayer.Process();

            HiddenLayer?.Process();

            OutputLayer.Process();

            return OutputLayer.Neurons.Select(n => n.Value).ToArray();
        }

        public void ClearDeltas()
        {
            foreach (Neuron hidden in HiddenLayer)
                hidden.Delta = 0;

            foreach (Neuron outn in OutputLayer)
                outn.Delta = 0;
        }
    }
}
