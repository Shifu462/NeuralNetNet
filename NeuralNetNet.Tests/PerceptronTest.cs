using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetNet.NeuralNetwork;

namespace NeuralNetNet.Tests
{
    [TestClass]
    public class PerceptronTest
    {
        [TestMethod]
        public void XorTest()
        {
            var xorTrainList = new List<TrainSet>();

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 1.0 }, Output = 1.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 0.0 }, Output = 1.0 });

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 0.0 }, Output = 0.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 1.0 }, Output = 0.0 });

            Perceptron net = new Perceptron(2, 1, hiddenCount: 1, hiddenSize: 4);

            net.Train(xorTrainList, 15000);

            foreach (TrainSet ts in xorTrainList)
            {
                double result = net.Predict(ts.Input)[0];
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

            Perceptron net = new Perceptron(2, 1, hiddenCount: 2, hiddenSize: 4);

            net.Train(equalTrainList, 15000, learningRate: 1);

            foreach (TrainSet ts in equalTrainList)
            {
                double result = net.Predict(ts.Input)[0];
                double roundedResult = Math.Round(result);

                Assert.AreEqual(ts.Output, roundedResult);
            }
        }
    }
}
