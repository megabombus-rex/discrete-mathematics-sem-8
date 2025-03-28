namespace FordFulkerson_Algorithm.Data
{
    public class Node
    {
        public int Number;
        public List<Edge> Edges;

        public Node(int number) {
            Number = number;
            Edges = new List<Edge>();
        }

    }
}
