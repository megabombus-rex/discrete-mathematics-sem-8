using GraphData;

namespace BellmanFordAlgorithm
{
    public class BellmanFordSolver
    {
        // returns a shortest path from a source to every target, empty if there is a negative-weight cycle
        public bool ShortestPathPossible(Graph graph, int source)
        {
            var distances = new List<int>(graph.GraphNodeCount);

            for (int i = 0; i < graph.GraphNodeCount; i++)
            {
                distances.Add(int.MaxValue);
            }
            distances[source] = 0;

            var negativeCycleNr = graph.GraphNodeCount - 1;

            for (int i = 0; i < graph.GraphNodeCount - 1; i++) 
            { 
                foreach (var edge in graph.Edges)
                {
                    // relaxation
                    if (distances[edge.StartNodeNr] != int.MaxValue && distances[edge.StartNodeNr] + edge.Weight < distances[edge.EndNodeNr])
                    {
                        if (i == negativeCycleNr)
                        {
                            return false;
                        }
                    }

                    distances[edge.EndNodeNr] = distances[edge.StartNodeNr] + edge.Weight;
                }
            }

            return true;
        }
    }
}
