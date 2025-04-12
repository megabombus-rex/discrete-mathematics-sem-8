namespace ShortestPathProblem_Algorithms.DataSavers
{
    public class BellmanFordSaver
    {
        public void SaveToCSV(string filename, string testTitle, List<string> lines)
        {
            // newline - test, date
            // node_count, temperature, final_edge_count, runtime_in_ms, path_was_found, negative_weights_included
            using (StreamWriter sw = new StreamWriter(filename, true))
            {
                sw.WriteLine(testTitle);

                foreach (string line in lines)
                {
                    sw.WriteLine(line);
                }
            }

        }


    }
}
