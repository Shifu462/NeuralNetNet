using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetNet.ActivationFunctions;
using NeuralNetNet.NeuralNetwork;

namespace NeuralNetNet.Tests
{
    [TestClass]
    public class FeedForwardTest
    {
        [TestMethod]
        public void XorTest()
        {
            var xorTrainList = new List<TrainSet>();

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 1.0 }, Output = 1.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 0.0 }, Output = 1.0 });

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 0.0 }, Output = 0.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 1.0 }, Output = 0.0 });

            Layer[] hiddenLayers = {
                new Layer(4, new Sigmoid())
            };

            FeedForward net = new FeedForward(new Layer(2, new Sigmoid()), 
                                            hiddenLayers, 
                                            new Layer(1, new Sigmoid()));

            net.Train(xorTrainList, 5000);

            foreach (TrainSet ts in xorTrainList)
            {
                double result = net.Handle(ts.Input)[0];
                double roundedResult = Math.Round(result);
                Console.WriteLine($"{ts.Input[0]} X {ts.Input[1]} = {result}");
                Assert.AreEqual(ts.Output, roundedResult);
            }
        }

        [TestMethod]
        public void EqualsTest()
        {
            var equalTrainList = new List<TrainSet>();

            // Add numbers
            for (int i = 0; i <= 10; i++)
            {
                for (int j = 0; j <= 10; j++)
                {
                    bool isEqual = i == j;

                    equalTrainList.Add(new TrainSet
                    {
                        Input = new double[] { i, j },
                        Output = isEqual ? 1.0 : 0.0
                    });
                }
            }

            FeedForward net = new FeedForward(new Layer(2, new Sigmoid()),
                                            new Layer[] { new Layer(4, new Sigmoid()) },
                                            new Layer(1, new Sigmoid()));

            net.Train(equalTrainList, 10000, learningRate: 1);

            foreach (TrainSet ts in equalTrainList)
            {
                double result = net.Handle(ts.Input)[0];
                double roundedResult = Math.Round(result);

                Assert.AreEqual(ts.Output, roundedResult);
            }
        }
    }
}
