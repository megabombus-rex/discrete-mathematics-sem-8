namespace FordFulkerson_Algorithm.Data
{
    public class Graph
    {
        private List<Node> _nodes;
        private List<Edge> _edges;

        public Graph()
        {
            _nodes = new List<Node>();
            _edges = new List<Edge>();
        }

        public void AddNode(Node node)
        {
            _nodes.Add(node);

            foreach (Edge edge in node.Edges)
            {
                _edges.Add(edge);
            }
        }

        public int GraphNodeCount { get { return _nodes.Count; } }

        public Node GetNodeByNumber(int nodeNumber)
        {
            return _nodes.First(x => x.Number == nodeNumber);
        }

        // O(n * v)
        public Graph GetGraphClone()
        {
            var clone = new Graph();
            var nodeListClone = new List<Node>();

            // O(n)
            foreach (Node node in _nodes)
            {
                var nodeClone = new Node(node.Number);
                nodeListClone.Add(nodeClone);
            }

            // O(v)
            foreach (var edge in _edges)
            {
                var edgeClone = new Edge();
                edgeClone.StartNode = edge.StartNode;
                edgeClone.EndNode = edge.EndNode;
                edgeClone.Capacity = edge.Capacity;

                var nodeClone = nodeListClone.First(x => x.Number == edgeClone.StartNode);
                nodeClone.Edges.Add(edgeClone);
            }

            foreach (var node in nodeListClone)
            {
                clone.AddNode(node);
            }

            return clone;
        }
    }
}
