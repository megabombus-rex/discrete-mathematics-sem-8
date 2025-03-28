using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm
{
    public class GraphCreator
    {
        public GraphCreator() { }

        public int[,] GenerateGraph(int nodeCount, double temperature, int weight)
        {
            var rng = new Random();
            int i = 0;

            int[,] graph = new int[nodeCount, nodeCount];
            

            for (i = 0; i < nodeCount; i++)
            {
                graph[i, i] = 0;
            }

            // a linked list of edges
            for (i = 0; i < nodeCount - 1; i++) 
            {
                int w = rng.Next(1, weight);
                graph[i, i + 1] = w;
            }

            int additionalEdges = (int)(Math.Round((double)nodeCount * temperature));

            //i = 0;
            //while (i < additionalEdges)
            //{
            //    var u = rng.Next(0, nodeCount - 1);
            //    var v = rng.Next(0, nodeCount);

            //    // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
            //    if (u != v && graph[u, v] == 0 && graph[v, u] == 0)
            //    {
            //        int w = rng.Next(1, weight);
            //        //Console.WriteLine($"Additional edge from node {u} to node {v} with flow {w}.");
            //        graph[u, v] = w;
            //        i++;
            //    }
            //}


            for (i = 0; i < additionalEdges; i++)
            {
                var u = rng.Next(0, nodeCount - 1);
                var v = rng.Next(0, nodeCount);

                // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
                if (u != v && graph[u, v] == 0 && graph[v, u] == 0)
                {
                    int w = rng.Next(1, weight);
                    //Console.WriteLine($"Additional edge from node {u} to node {v} with flow {w}.");
                    graph[u, v] = w;
                    //i++;
                }
            }

            return graph;
        }


        public int[][] GenerateGraphNotJagged(int nodeCount, double temperature, int weight)
        {
            var rng = new Random();
            int i = 0;

            int[][] graph = new int[nodeCount][];
            for (i = 0; i < nodeCount; i++)
            {
                graph[i] = new int[nodeCount];
            }

            for (i = 0; i < nodeCount; i++)
            {
                graph[i][i] = 0;
            }

            // a linked list of edges
            for (i = 0; i < nodeCount - 1; i++)
            {
                int w = rng.Next(1, weight);
                graph[i][i + 1] = w;
            }

            int additionalEdges = (int)(Math.Round((double)nodeCount * temperature));

            //i = 0;
            //while (i < additionalEdges)
            //{
            //    var u = rng.Next(0, nodeCount - 1);
            //    var v = rng.Next(0, nodeCount);

            //    // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
            //    if (u != v && graph[u][v] == 0 && graph[v][u] == 0)
            //    {
            //        int w = rng.Next(1, weight);
            //        //Console.WriteLine($"Additional edge from node {u} to node {v} with flow {w}.");
            //        graph[u][v] = w;
            //        i++;
            //    }

            //}

            Console.WriteLine($"Additional edges: {additionalEdges}");

            for (i = 0; i < additionalEdges; i++)
            {
                var u = rng.Next(0, nodeCount - 1);
                var v = rng.Next(0, nodeCount);

                // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
                if (u != v && graph[u][v] == 0 && graph[v][u] == 0)
                {
                    int w = rng.Next(1, weight);
                    //Console.WriteLine($"Additional edge from node {u} to node {v} with flow {w}.");
                    graph[u][v] = w;
                    //i++;
                }
            }

            return graph;
        }

        public Graph GenerateGraphStructure(int nodeCount, double temperature, int weight)
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
                edge.StartNode = i; 
                edge.EndNode = i + 1;
                edge.Capacity = w;

                nodeList[i].Edges.Add(edge);
            }

            int additionalEdges = (int)Math.Round((double)nodeCount * temperature);


            Console.WriteLine($"Additional edges: {additionalEdges}");

            for (i = 0; i < additionalEdges; i++)
            {
                var u = rng.Next(0, nodeCount - 1);
                var v = rng.Next(0, nodeCount);

                // graph[u, v] == 0 && graph[v,u] == 0 -> no bi-directional graphs
                if (u != v && !nodeList[u].Edges.Any(x => x.EndNode == v))// && !nodeList[v].Edges.Any(x => x.EndEdge == u)) - this may be unnecessary
                {
                    int w = rng.Next(1, weight);
                    var edge = new Edge();
                    edge.StartNode = u;
                    edge.EndNode = v;
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

        public void DrawGraph(int[,] graph, int size)
        {
            for (int i = 0; i < size; i++)
            {
                string nodes = "[";
                for (int j = 0; j < size - 1; j++)
                {
                    nodes += $"\t{graph[i, j]},";
                }
                nodes += $"\t{graph[i,size - 1]}]";

                Console.WriteLine($"Line {i}: {nodes}");
            }
        }
    }

}
