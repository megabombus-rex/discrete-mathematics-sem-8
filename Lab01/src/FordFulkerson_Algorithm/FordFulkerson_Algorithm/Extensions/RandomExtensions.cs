namespace FordFulkerson_Algorithm.Extensions
{
    public static class RandomExtensions
    {
        // https://stackoverflow.com/questions/1064901/random-number-between-2-double-numbers
        public static double RandomDouble(this Random rng, double minimum, double maximum)
        {
            return rng.NextDouble() * (maximum - minimum) + minimum;
        }
    }
}
