using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm
{
    public class GraphCreator
    {
        public GraphCreator() { }

        /// <summary>
        /// Randomly creates one directed graph with flows set between given limits.
        /// </summary>
        /// <param name="nodeCount">How many nodes there is.</param>
        /// <param name="minFlow">Minimal max flow for an edge.</param>
        /// <param name="maxFlow">Maximum max flow for an edge.</param>
        /// <returns>A random one-directed graph.</returns>
        public Graph CreateRandomOneDirectedGraphFordFulkerson(int nodeCount, int minFlow, int maxFlow) 
        {     
            if (nodeCount < 1)
            {
                throw new ArgumentException("Not enough nodes to create a graph.");
            }
            if (minFlow < 1)
            {
                throw new ArgumentException($"There cannot be an edge with flow lesser than 1.");
            }
            if (maxFlow < 1)
            {
                throw new ArgumentException($"There cannot be an edge with flow lesser than 1.");
            }
            if (maxFlow < minFlow)
            {
                throw new ArgumentException($"Max flow cannot be smaller than min flow.");
            }

            var graph = new Graph();
            var rng = new Random();
            graph.Source = new Node(0); // node 0
            graph.Sink = new Node(nodeCount + 1); // max node

            for (int i = 0; i < nodeCount; i++) {
                graph.Nodes.Add(new Node(i + 1));
            }
            
            var sourceLinkedNodesNumber = rng.Next(1, nodeCount);

            for (int i = 0; i < sourceLinkedNodesNumber; i++) 
            {
                
            }
            return new Graph(); 
        }
    }
}
