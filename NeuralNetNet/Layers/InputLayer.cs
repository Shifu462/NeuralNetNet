
namespace NeuralNetNet.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(int neuronsCount) : base(neuronsCount, null)
        {

        }


        public override void Process()
        {
            // Normalize: 1 / inputs
        }
    }
}