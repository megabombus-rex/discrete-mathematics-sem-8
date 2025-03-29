using FordFulkerson_Algorithm.Data;
using FordFulkerson_Algorithm.FileEditors;

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
            Console.WriteLine($"A graph of size {result.GraphSize} generated with temperature {result.Temperature} has a max flow calculated as {result.MaxFlow}. It took {result.Duration.TotalMilliseconds}ms.");
        }
    }
}
