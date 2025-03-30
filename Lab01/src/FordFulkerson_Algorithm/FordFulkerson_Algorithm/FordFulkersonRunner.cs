using FordFulkerson_Algorithm.Data;
using FordFulkerson_Algorithm.FileEditors;
using System.Runtime;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace FordFulkerson_Algorithm
{
    public class FordFulkersonRunner
    {
        private readonly GraphCreator _graphCreator;
        private readonly MaxFlowFordFulkerson _fordFulkerson;
        private readonly GraphToFileSaver _graphToFileSaver;

        public FordFulkersonRunner()
        {
            _graphCreator = new GraphCreator();
            _fordFulkerson = new MaxFlowFordFulkerson();
            _graphToFileSaver = new GraphToFileSaver();
        }

        public Result TestAndSaveGraph(Graph graph, double temperature)
        {
            //if (graph.GraphNodeCount > 5000)
            //{
            //    GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            //    GC.Collect();
            //}
            var start = DateTime.UtcNow;
            var maxFlow = _fordFulkerson.FordFulkerson(graph, 0, graph.GraphNodeCount - 1);
            var duration = DateTime.UtcNow - start;

            var date = (start.ToShortDateString() + start.ToShortTimeString()).Replace(':', '_').Replace(' ', '_');

            var filepath = $"{Environment.CurrentDirectory}\\..\\..\\..\\SavedGraphs\\Graph_{graph.GraphNodeCount}_t_{(int)temperature}_{date}";//string.Format("{0}\\{1}\\{2}", Environment.CurrentDirectory, "..\\..\\..\\SavedGraphs", $"graph");

            _graphToFileSaver.SaveGraph(graph, filepath);

            return new Result(duration, graph.GraphNodeCount, temperature, maxFlow);
        }

        public Result GenerateGraphAndCalculateFlow(int graphSize, double temperature, int maxFlowPerEdge) 
        {
            //var graph = _graphCreator.GenerateGraph(graphSize, temperature, maxFlowPerEdge);
            //var graph = _graphCreator.GenerateGraphNotJagged(graphSize, temperature, maxFlowPerEdge);
            var graph = _graphCreator.GenerateGraphStructure(graphSize, temperature, maxFlowPerEdge);


            var start = DateTime.UtcNow;
            var maxFlow = _fordFulkerson.FordFulkerson(graph, 0, graphSize - 1);
            var duration = DateTime.UtcNow - start;

            return new Result(duration, graphSize, temperature, maxFlow);
        }

        public void ShowResult(Result result)
        {
            var start = DateTime.UtcNow;
            var date = (start.ToShortDateString() + start.ToShortTimeString()).Replace(':', '_').Replace(' ', '_');
            var data = $"A graph of size {result.GraphSize} generated with temperature {result.Temperature} has a max flow calculated as {result.MaxFlow}. It took {result.Duration.TotalMilliseconds}ms. At {date}.";
            //Console.WriteLine(data);
            var filepath = $"{Environment.CurrentDirectory}\\..\\..\\..\\Results\\GraphRes_{result.GraphSize}_t_{result.Temperature}";//string.Format("{0}\\{1}\\{2}", Environment.CurrentDirectory, "..\\..\\..\\SavedGraphs", $"graph");

            using (StreamWriter sw = new StreamWriter(filepath, true)) 
            {
                sw.WriteLine(data);
            }
        }
    }
}
