namespace FordFulkerson_Algorithm.Data
{
    public class Graph
    {
        public List<Node> Nodes { get; set; }

        public Node Sink { get; set; }
        public Node Source { get; set; }

        public Graph()
        {
            Nodes = new List<Node>();
            Sink = new Node();
            Source = new Node();
        }
    }
}
