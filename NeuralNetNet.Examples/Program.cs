﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetNet.Examples
{
    class Program
    {
        static void Main(string[] args)
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

            Perceptron net = new Perceptron(2, 1, hiddenCount: 12, hiddenSize: 12);

            net.Train(equalTrainList, 100000, learningRate: 1);

            Console.Clear();

            Console.WriteLine("Expected 1:");
            foreach (TrainSet ts in equalTrainList.Where(t => t.Output == 1.0))
                Console.WriteLine(net.Predict(ts.Input)[0]);

            Console.WriteLine("--\nOut of range:");
            Console.WriteLine(net.Predict(11, 11)[0]);
            Console.WriteLine(net.Predict(12, 12)[0]);
            Console.WriteLine(net.Predict(13, 13)[0]);
            Console.WriteLine(net.Predict(14, 14)[0]);
            Console.WriteLine(net.Predict(15, 15)[0]);


            Console.WriteLine("\nExpected 0:");
            foreach (TrainSet ts in equalTrainList.Where(t => t.Output == 0.0))
                Console.WriteLine(net.Predict(ts.Input)[0]);


            Console.WriteLine("--\nOut of range:");
            Console.WriteLine(net.Predict(11, 10)[0]);
            Console.WriteLine(net.Predict(12, 11)[0]);
            Console.WriteLine(net.Predict(13, 10)[0]);

            Console.WriteLine(net.Predict(20, 19)[0]);
            Console.WriteLine(net.Predict(20, 18)[0]);

            Console.ReadKey();
        }
    }
}
