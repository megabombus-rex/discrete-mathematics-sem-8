using GraphData;

namespace FloydMarshallAlgorithm
{
    public class FloydWarshallSolver
    {
        public bool ShortestPathArrayWithNoNegativeLoops(Graph graph)
        {
            var n = graph.GraphNodeCount;
            var distances = new int[n][];
            var infinity = (int)10e8;

            for (int i = 0; i < n; i++)
            {
                var row = new int[n];
                for (int j = 0; j < n; j++)
                {
                    row[j] = infinity;
                    Console.WriteLine($"If the value is {infinity} then it is not reachable at the end.");
                }
                distances[i] = row;
            }

            foreach (var edge in graph.Edges) 
            {
                distances[edge.StartNodeNr][edge.EndNodeNr] = edge.Weight;
            }

            // O(n^3)
            for (int k = 0; k < n; k++)
            {
                // O(n^2)
                for (int i = 0; i < n; i++)
                {
                    // O(n)
                    for (int j = 0; j < n; j++)
                    {
                        if (distances[i][j] > distances[i][k] + distances[k][j])
                        {
                            distances[i][j] = distances[i][k] + distances[k][j];
                        }
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                if (distances[i][i] < 0)
                {
                    return false;
                }
            }

            return true;
        }

        public int[][] ShortestPathArray(Graph graph)
        {
            var n = graph.GraphNodeCount;
            var distances = new int[n][];
            var infinity = (int)10e8;

            for (int i = 0; i < n; i++)
            {
                var row = new int[n];
                for (int j = 0; j < n; j++)
                {
                    row[j] = infinity;
                    Console.WriteLine($"If the value is {infinity} then it is not reachable at the end.");
                }
                distances[i] = row;
            }

            foreach (var edge in graph.Edges)
            {
                distances[edge.StartNodeNr][edge.EndNodeNr] = edge.Weight;
            }

            // O(n^3)
            for (int k = 0; k < n; k++)
            {
                // O(n^2)
                for (int i = 0; i < n; i++)
                {
                    // O(n)
                    for (int j = 0; j < n; j++)
                    {
                        if (distances[i][j] > distances[i][k] + distances[k][j])
                        {
                            distances[i][j] = distances[i][k] + distances[k][j];
                        }
                    }
                }
            }

            return distances;
        }
    }
}
