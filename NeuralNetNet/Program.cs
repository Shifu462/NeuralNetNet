using System;
using System.Collections.Generic;

namespace NeuralNetNet
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();

            var trainList = new List<XorTrainSet>();

            trainList.Add(new XorTrainSet() { Input = new double[] { 0, 1.0 }, Output = 1.0 });
            trainList.Add(new XorTrainSet() { Input = new double[] { 1.0, 0 }, Output = 1.0 });

            trainList.Add(new XorTrainSet() { Input = new double[] { 0, 0 }, Output = 0 });
            trainList.Add(new XorTrainSet() { Input = new double[] { 1, 1 }, Output = 0 });

            NeuralNetwork net = new NeuralNetwork(2, 1, hiddenCount: 1, hiddenSize: 5);

            net.Train(trainList, 10000);

            Console.WriteLine("Expected 0:");
            Console.WriteLine(net.Predict(0, 0)[0]);
            Console.WriteLine(net.Predict(1, 1)[0]);

            Console.WriteLine("\nExpected 1:");
            Console.WriteLine(net.Predict(0, 1)[0]);
            Console.WriteLine(net.Predict(1, 0)[0]);

            Console.WriteLine();

            Console.WriteLine("Weights @ Output Layer:");
            foreach (Neuron n in net.OutputLayer.Neurons)
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
