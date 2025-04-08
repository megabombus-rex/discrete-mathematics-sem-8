using BellmanFordAlgorithm;
using GraphData;

namespace ShortestPathProblem_Algorithms.Tests.BellmanFord
{
    public class BFSolverTest_1
    {
        public void Run()
        {
            BellmanFordSolver solverBF = new BellmanFordSolver();

            var graph = new Graph();

            var nodeA = new Node(0);
            var nodeB = new Node(1);
            var nodeC = new Node(2);
            var nodeD = new Node(3);
            var nodeE = new Node(4);

            var edgeAB = new Edge() { StartNodeNr = 0, EndNodeNr = 1, Weight = 5 };
            var edgeBC = new Edge() { StartNodeNr = 1, EndNodeNr = 2, Weight = 1 };
            var edgeBD = new Edge() { StartNodeNr = 1, EndNodeNr = 3, Weight = 2 };
            var edgeCE = new Edge() { StartNodeNr = 2, EndNodeNr = 4, Weight = 1 };
            var edgeED = new Edge() { StartNodeNr = 4, EndNodeNr = 3, Weight = -1 };

            nodeA.Edges.Add(edgeAB);
            nodeB.Edges.Add(edgeBC);
            nodeB.Edges.Add(edgeBD);
            nodeC.Edges.Add(edgeCE);
            nodeE.Edges.Add(edgeED);

            graph.AddNode(nodeA);
            graph.AddNode(nodeB);
            graph.AddNode(nodeC);
            graph.AddNode(nodeD);
            graph.AddNode(nodeE);

            Console.WriteLine(solverBF.ShortestPathPossible(graph, 0));
        }
    }
}
