using BenchmarkDotNet.Attributes;
using FordFulkerson_Algorithm.Data;

namespace FordFulkerson_Algorithm.Benchmarks
{
    public class FordFulkersonBenchmark
    {
        private readonly Graph _graph;
        private readonly GraphCreator _creator;
        private readonly MaxFlowFordFulkerson _fordFulkerson;

        public FordFulkersonBenchmark()
        {
            _creator = new GraphCreator();
            _fordFulkerson = new MaxFlowFordFulkerson();
            _graph = _creator.GenerateGraphStructure(50000, 1.0, 50);
        }

        [Benchmark]
        public int CheckFordFulkerson()
        {
            return _fordFulkerson.FordFulkerson(_graph, 0, _graph.Nodes.Count - 1);
        }
    }
}
