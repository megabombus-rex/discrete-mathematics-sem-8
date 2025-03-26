﻿using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm
{
    public class MaxFlowFordFulkerson
    {
        public MaxFlowFordFulkerson()
        {
            
        }

        //returns true if the residiual graph contains path from source to sink
        bool BFS(int[,] graph, int source, int sink, int[] parent, int size)
        {
            // Create a visited array and mark all vertices as not visited
            bool[] visited = new bool[size];
            for (int i = 0; i < size; i++)
            {
                visited[i] = false;
            }

            // Create a queue, enqueue source vertex and mark source vertex as visited
            Queue<int> queue = new Queue<int>();
            queue.Enqueue(source);
            visited[source] = true;
            parent[source] = -1;

            // Standard BFS Loop
            while (queue.Count != 0)
            {
                int currentNode = queue.Dequeue(); //queue[0];

                for (int targetNode = 0; targetNode < size; targetNode++)
                {
                    if (visited[targetNode] == false && graph[currentNode, targetNode] > 0)
                    {
                        // If we find a connection to the sink
                        // node, then there is no point in BFS
                        // anymore We just have to set its parent
                        // and can return true
                        if (targetNode == sink)
                        {
                            parent[targetNode] = currentNode;
                            return true;
                        }
                        queue.Enqueue(targetNode);
                        parent[targetNode] = currentNode;
                        visited[targetNode] = true;
                    }
                }
            }
            return false;
        }

        public int FordFulkerson(int[,] graph, int source, int sink, int size)
        {
            int u, v;

            // Create a residual graph and fill
            // the residual graph with given
            // capacities in the original graph as
            // residual capacities in residual graph

            // Residual graph where rGraph[i,j]
            // indicates residual capacity of
            // edge from i to j (if there is an
            // edge. If rGraph[i,j] is 0, then
            // there is not)
            int[,] rGraph = new int[size, size];

            for (u = 0; u < size; u++)
                for (v = 0; v < size; v++)
                    rGraph[u, v] = graph[u, v];

            // This array is filled by BFS and to store path
            int[] parent = new int[size];

            int max_flow = 0; // There is no flow initially

            // Augment the flow while there is path from source
            // to sink
            while (BFS(rGraph, source, sink, parent, size))
            {
                // Find minimum residual capacity of the edges
                // along the path filled by BFS. Or we can say
                // find the maximum flow through the path found.
                int pathFlow = int.MaxValue;
                for (v = sink; v != source; v = parent[v])
                {
                    u = parent[v];
                    pathFlow
                        = Math.Min(pathFlow, rGraph[u, v]);
                }

                // update residual capacities of the edges and
                // reverse edges along the path
                for (v = sink; v != source; v = parent[v])
                {
                    u = parent[v];
                    rGraph[u, v] -= pathFlow;
                    rGraph[v, u] += pathFlow;
                }

                // Add path flow to overall flow
                max_flow += pathFlow;
            }

            // Return the overall flow
            return max_flow;
        }
    }
}
