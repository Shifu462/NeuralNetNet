using System;

namespace NeuralNetNet
{
    /// <summary>
    /// Extension class for math functions.
    /// </summary>
    internal class MathFunctions
    {
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
