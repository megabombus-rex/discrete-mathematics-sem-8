using GraphData;
using MathNet.Numerics.LinearAlgebra;

namespace FloydMarshallAlgorithm
{
    public class FloydWarshallSolver
    {
        public Matrix<int> ShortestPathArray(Graph graph)
        {
            var matrix = graph.TranslateGraphToAdjacencyMatrixPlus();

            return ShortestPathPossible(matrix);
        }

        public Matrix<int> ShortestPathPossible(Matrix<int> graph)
        {
            var n = graph.RowCount;
            var D0 = graph.Clone();

            // D0 is D(1 - 1)
            for (int k = 1; k < n; k++)
            {
                var Dk = D0.Clone();
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                    }
                }
            }

            // wrong
            return D0.Clone();
        }

        private int CalculateDistanceRecursive(Matrix<int> adjacencyMatrix, int k, int i, int j)
        {
            if (k == 0)
            {
               // weight
               return adjacencyMatrix[i, j];
            }
        }
    }
}
