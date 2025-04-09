using GraphData;

namespace BellmanFordAlgorithm
{
    public class BellmanFordSolver
    {
        // returns a shortest path from a source to every target, empty if there is a negative-weight cycle
        public bool ShortestPathPossible(Graph graph, int source)
        {
            var distances = new int[graph.GraphNodeCount];
            var predecesors = new int?[graph.GraphNodeCount];
            var infinity = (int)10e8;

            Console.WriteLine($"If the value is {infinity} then it is not reachable at the end.");

            for (int i = 0; i < graph.GraphNodeCount; i++)
            {
                distances[i] = infinity;
                predecesors[i] = null;
            }
            distances[source] = 0;


            for (int i = 0; i < graph.GraphNodeCount; i++)
            {
                // relax each edge
                foreach (var edge in graph.Edges)
                {
                    if (distances[edge.EndNodeNr] > distances[edge.StartNodeNr] + edge.Weight)
                    {
                        distances[edge.EndNodeNr] = distances[edge.StartNodeNr] + edge.Weight;
                        predecesors[edge.EndNodeNr] = edge.StartNodeNr;
                    }
                }
            }

            foreach (var edge in graph.Edges)
            {
                if (distances[edge.EndNodeNr] > distances[edge.StartNodeNr] + edge.Weight)
                {
                    return false;
                }
            }

            Console.WriteLine($"Distances: [{string.Join("|", distances)}]");
            return true;
        }
    }
}
