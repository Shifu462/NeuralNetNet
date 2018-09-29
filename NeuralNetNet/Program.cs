using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetNet
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();

            var xorTrainList = new List<TrainSet>();

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 1.0 }, Output = 1.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 0.0 }, Output = 1.0 });

            xorTrainList.Add(new TrainSet { Input = new double[] { 0.0, 0.0 }, Output = 0.0 });
            xorTrainList.Add(new TrainSet { Input = new double[] { 1.0, 1.0 }, Output = 0.0 });

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

            NeuralNetwork net = new NeuralNetwork(2, 1, hiddenCount: 1, hiddenSize: 4);

            net.Train(equalTrainList, 15000, learningRate: 1);

            Console.WriteLine("Expected 1:");
            Console.WriteLine(net.Predict(0, 0)[0]);
            Console.WriteLine(net.Predict(1, 1)[0]);
            Console.WriteLine(net.Predict(2, 2)[0]);
            Console.WriteLine(net.Predict(3, 3)[0]);
            Console.WriteLine(net.Predict(4, 4)[0]);
            Console.WriteLine(net.Predict(5, 5)[0]);
            Console.WriteLine(net.Predict(6, 6)[0]);
            Console.WriteLine(net.Predict(7, 7)[0]);
            Console.WriteLine(net.Predict(8, 8)[0]);
            Console.WriteLine(net.Predict(9, 9)[0]);
            Console.WriteLine(net.Predict(10, 10)[0]);

            Console.WriteLine("\nExpected 0:");
            Console.WriteLine(net.Predict(1, 0)[0]);
            Console.WriteLine(net.Predict(0, 1)[0]);
            Console.WriteLine(net.Predict(1, 2)[0]);
            Console.WriteLine(net.Predict(1, 3)[0]);
            Console.WriteLine(net.Predict(2, 3)[0]);
            Console.WriteLine(net.Predict(2, 5)[0]);
            Console.WriteLine(net.Predict(4, 3)[0]);
            Console.WriteLine(net.Predict(4, 8)[0]);

            Console.WriteLine();

            Console.WriteLine("Weights @ Output Layer:");
            foreach (Neuron n in net.OutputLayer)
            {
                foreach (double w in n.SynapsesWeights)
                {
                    Console.WriteLine(w.ToString());
                }
                Console.WriteLine(@"-------");
            }
        }
    }
}
