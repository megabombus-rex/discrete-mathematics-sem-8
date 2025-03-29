using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm.FileEditors
{
    public class GraphToFileSaver
    {
        public void SaveGraph(Graph graph, string filename)
        {
            var filePath = Path.GetDirectoryName(filename);
            (new DirectoryInfo(Path.GetFullPath(filePath))).Create();
            
            using (StreamWriter sw = new StreamWriter(filename, false))
            {
                foreach (var node in graph.Nodes)
                {
                    sw.WriteLine(node.Number);
                }

                foreach (var node in graph.Nodes)
                {
                    foreach (var edge in node.Edges)
                    {
                        sw.WriteLine($"{node.Number} {edge.EndNode} {edge.Capacity}");
                    }
                }

            }
        }
    }
}
