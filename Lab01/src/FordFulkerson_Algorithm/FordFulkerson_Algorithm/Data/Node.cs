namespace FordFulkerson_Algorithm.Data
{
    public class Node
    {
        public int Id { get; set; }
        public List<Edge> Edges { get; set; }

        public Node()
        {
            Id = 0;
            Edges = new List<Edge>();
        }

        public Node(int id)
        {
            Id = id;
            Edges = new List<Edge>();
        }
    }
}
