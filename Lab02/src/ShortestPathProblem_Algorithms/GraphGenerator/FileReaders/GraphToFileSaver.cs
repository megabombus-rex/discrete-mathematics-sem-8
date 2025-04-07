using GraphData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GraphGenerator.FileReaders
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
                        sw.WriteLine($"{node.Number} {edge.EndNodeNr} {edge.Weight}");
                    }
                }

            }
        }
    }
}
