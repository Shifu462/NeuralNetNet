using System;

namespace NeuralNetNet.ActivationFunctions
{
    public class Sigmoid : IActivation
    {
        public double Activate(double value, bool derivative)
        {
            if (derivative) // actually sigmoid(value) * (1 -sigmoid(value))
                return value * (1 - value); // but the value is already sigmoided.

            return ApproxSigmoid(value);
        }

        protected double ApproxSigmoid(double x)
        {
            double exp = ApproxExp(-x);
            exp = 1.0 / (1.0 + exp);
            return exp;
        }

        protected double ApproxExp(double x)
        {
            long tmp = (long)(1512775 * x + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }
    }
}
