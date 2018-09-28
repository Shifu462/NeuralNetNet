namespace NeuralNetNet.Layers
{
    public class HiddenLayer : Layer
    {
        public Layer NextLayer { get; set; }

        public HiddenLayer(int neuronsCount, Layer prev) : base(neuronsCount, prev)
        {

        }
    }
}