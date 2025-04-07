using GraphData;

namespace GraphTraversalAlgorithms
{
    public class BFS
    {
        bool CanTraverseGraph(Graph graph, int source, int target, int[] parent)
        {
            bool[] visited = new bool[graph.GraphNodeCount];
            for (int i = 0; i < graph.GraphNodeCount; i++)
            {
                visited[i] = false;
            }

            Queue<Node> queue = new Queue<Node>();
            queue.Enqueue(graph.GetNodeByNumber(source));
            visited[source] = true;
            parent[source] = -1;


            // Standard BFS Loop
            while (queue.Count != 0)
            {
                var currentNode = queue.Dequeue(); //queue[0];

                for (int targetNode = 0; targetNode < graph.GraphNodeCount; targetNode++)
                {
                    if (visited[targetNode] == false && currentNode.Edges.Any(x => x.EndNodeNr == targetNode && x.Capacity > 0))
                    {
                        if (targetNode == target)
                        {
                            parent[targetNode] = currentNode.Number;
                            return true;
                        }
                        queue.Enqueue(graph.GetNodeByNumber(targetNode));
                        parent[targetNode] = currentNode.Number;
                        visited[targetNode] = true;
                    }
                }
            }

            return false;
        }
    }
}
