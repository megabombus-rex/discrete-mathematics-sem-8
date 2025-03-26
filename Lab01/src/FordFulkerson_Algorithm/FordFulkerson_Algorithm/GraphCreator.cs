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

            //Iterate through the list and for each element, make the probability of selection = (number needed) / (number left)
            //So if you had 40 items, the first would have a 5 / 40 chance of being selected.
            //If it is, the next has a 4 / 39 chance, otherwise it has a 5 / 39 chance.By the time you get to the end you will have your 5 items, and often you'll have all of them before that.

            var sourceLinkedNodesNumberLeft = rng.Next(1, nodeCount);
            var sourceNodesNumbers = new List<int>();

            while (graph.Source.Edges.Count < sourceLinkedNodesNumberLeft)
            {
                var prob = (double)sourceLinkedNodesNumberLeft / (double)nodeCount;
                //var prob = rng.RandomDouble(0.0, (double)nodeCount) / sourceLinkedNodesNumberLeft;
                //if (prob < )

                //var nodeNr = rng.Next(1, nodeCount);

            }

            return new Graph(); 
        }
    }
}
