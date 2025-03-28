using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm
{
    public class FordFulkersonRunner
    {
        private readonly GraphCreator _graphCreator;
        private readonly MaxFlowFordFulkerson _fordFulkerson;

        public FordFulkersonRunner()
        {
            _graphCreator = new GraphCreator();
            _fordFulkerson = new MaxFlowFordFulkerson();
        }

        public Result GenerateGraphAndCalculateFlow(int graphSize, double temperature, int maxFlowPerEdge) 
        {
            //var graph = _graphCreator.GenerateGraph(graphSize, temperature, maxFlowPerEdge);
            var graph = _graphCreator.GenerateGraphNotJagged(graphSize, temperature, maxFlowPerEdge);


            var start = DateTime.UtcNow;
            var maxFlow = _fordFulkerson.FordFulkerson(graph, 0, graphSize - 1, graphSize);
            var duration = DateTime.UtcNow - start;

            return new Result(duration, graphSize, temperature, maxFlow);
        }

        public void ShowResult(Result result)
        {
            Console.WriteLine($"A graph of size {result.GraphSize} generated with temperature {result.Temperature} has a max flow calculated as {result.MaxFlow}. It took {result.Duration.TotalMilliseconds}ms.");
        }
    }
}
