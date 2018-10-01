using System;

namespace NeuralNetNet
{
    internal class MathFunctions
    {
        public static double ApproxSigmoid(double x)
        {
            double exp = ApproxExp(-x);
            exp = 1.0 / (1.0 + exp);
            return exp;
        }

        protected static double ApproxExp(double x)
        {
            long tmp = (long)(1512775 * x + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }

        public static double MSE(double[] calc, double[] actual)
        {
            if (calc.Length != actual.Length)
                throw new ArgumentException("Calc.Length is not equal to Actual.Length");

            double sum = 0;

            for (int i = 0; i < calc.Length; i++)
            {
                double error = calc[i] - actual[i];
                sum += error * error;
            }

            return sum / actual.Length;
        }

        public static double Normalize(double value)
        {
            if (value == 0)
                return 0;
            return 1.0 / value;
        }
    }
}
