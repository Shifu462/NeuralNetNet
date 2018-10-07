using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetNet.NeuralNetwork;

namespace NeuralNetNet.Tests
{
    [TestClass]
    public class KohonenTest
    {
        [TestMethod]
        public void KohonenNetworkIsUntested()
        {
            KohonenNetwork net = new KohonenNetwork(2, 2);

            for (int ep = 0; ep < 100; ep++)
            {
                net.Study(new int[] { 0, 0 }, 1, 1);
                net.Study(new int[] { 1, 1 }, 1, 1);
                net.Study(new int[] { 1, 0 }, 0, 1);
                net.Study(new int[] { 0, 1 }, 0, 1);
            }
            

            var result = new int[]
            {
                net.Handle(0, 0),
                net.Handle(1, 1),
                net.Handle(0, 1),
                net.Handle(1, 0),
            };
        }
    }
}
