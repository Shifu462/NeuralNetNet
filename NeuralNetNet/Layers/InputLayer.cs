
namespace NeuralNetNet.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(int neuronsCount) : base(neuronsCount, null)
        {

        }


        public override void Process()
        {
            foreach (Neuron n in this.Neurons)
                n.Value = Normalize(n.Value);
        }

        public double Normalize(double value)
        {
            if (value == 0)
                return 0;
            return 1.0 / value;
        }
    }
}