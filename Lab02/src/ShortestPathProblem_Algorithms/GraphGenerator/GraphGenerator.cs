using GraphData;

namespace GraphGenerator
{
    public class GraphGenerator
    {
        public Graph GenerateGraphWithNegativeWeights(int nodeCount, double temperature, int weight)
        {
            var graph = new Graph();
            var rng = new Random();
            int i = 0;

            var nodeList = new List<Node>();

            for (i = 0; i < nodeCount; i++)
            {
                var node = new Node(i);
                nodeList.Add(node);
            }

            // a linked list of edges
            for (i = 0; i < nodeCount - 1; i++)
            {
                int w = rng.Next(-weight, weight);
                var edge = new Edge();
                edge.StartNodeNr = i;
                edge.EndNodeNr = i + 1;
                edge.Capacity = w;

                nodeList[i].Edges.Add(edge);
            }

            int additionalEdges = (int)Math.Round((double)nodeCount * temperature);


            //Console.WriteLine($"Additional edges: {additionalEdges}");

            for (i = 0; i < additionalEdges; i++)
            {
                var u = rng.Next(0, nodeCount - 1);
                var v = rng.Next(0, nodeCount);

                // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
                if (u != v && !nodeList[u].Edges.Any(x => x.EndNodeNr == v))// && !nodeList[v].Edges.Any(x => x.EndEdge == u)) - this may be unnecessary
                {
                    int w = rng.Next(1, weight);
                    var edge = new Edge();
                    edge.StartNodeNr = u;
                    edge.EndNodeNr = v;
                    edge.Capacity = w;

                    nodeList[u].Edges.Add(edge);
                }
            }


            foreach (var node in nodeList)
            {
                graph.AddNode(node);
            }

            return graph;
        }

        public Graph GenerateGraphOnlyNonNegativeWeights(int nodeCount, double temperature, int weight)
        {
            var graph = new Graph();
            var rng = new Random();
            int i = 0;

            var nodeList = new List<Node>();

            for (i = 0; i < nodeCount; i++)
            {
                var node = new Node(i);
                nodeList.Add(node);
            }

            // a linked list of edges
            for (i = 0; i < nodeCount - 1; i++)
            {
                int w = rng.Next(1, weight);
                var edge = new Edge();
                edge.StartNodeNr = i;
                edge.EndNodeNr = i + 1;
                edge.Capacity = w;

                nodeList[i].Edges.Add(edge);
            }

            int additionalEdges = (int)Math.Round((double)nodeCount * temperature);


            //Console.WriteLine($"Additional edges: {additionalEdges}");

            for (i = 0; i < additionalEdges; i++)
            {
                var u = rng.Next(0, nodeCount - 1);
                var v = rng.Next(0, nodeCount);

                // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
                if (u != v && !nodeList[u].Edges.Any(x => x.EndNodeNr == v))// && !nodeList[v].Edges.Any(x => x.EndEdge == u)) - this may be unnecessary
                {
                    int w = rng.Next(1, weight);
                    var edge = new Edge();
                    edge.StartNodeNr = u;
                    edge.EndNodeNr = v;
                    edge.Capacity = w;

                    nodeList[u].Edges.Add(edge);
                }
            }


            foreach (var node in nodeList)
            {
                graph.AddNode(node);
            }

            return graph;
        }
    }
}
