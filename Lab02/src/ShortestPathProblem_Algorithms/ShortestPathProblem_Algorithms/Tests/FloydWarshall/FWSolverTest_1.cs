using FloydMarshallAlgorithm;
using ShortestPathProblem_Algorithms.DataSavers;

namespace ShortestPathProblem_Algorithms.Tests.FloydWarshall
{
    public class FWSolverTest_1
    {
        public void Run()
        {
            var solverFW = new FloydWarshallSolver();

            var temperatures = new double[] { 0.5, 1.0, 2.0 };

            var graphSizes = new int[] { 10, 100, 500, 1000, 2000 };

            var weights_neg = new int[] { 30, 50, 100 };
            var weights_pos = new int[] { 30, 50, 100 };

            var filepath = $"{Environment.CurrentDirectory}\\..\\..\\..\\Results\\FloydWarshall\\FloydWarshallTests.csv";
            var title = $"FWSolverTest x100 iterations, 5k V, 20k E max, {DateTime.Now}";

            // newline - test, date
            // node_count, temperature, final_edge_count, runtime_in_ms, path_was_found, negative_weights_included

            var iterationsLimit = 10;

            var taskCount = 10;
            var iterationsPerTask = iterationsLimit / taskCount;
            var resultList = new List<string>((graphSizes.Length * temperatures.Length * (weights_neg.Length + weights_pos.Length) * iterationsLimit));

            var tasks = new Task[taskCount];

            for (int t = 0; t < taskCount; t++)
            {
                var taskNr = t;
                int iter = iterationsPerTask;
                tasks[t] = Task.Run(() =>
                {
                    Console.WriteLine($"Starting task {taskNr}.");
                    var graphCreator = new GraphGenerator.GraphCreator();
                    //Console.WriteLine($"Running task {t}.");
                    int iter = iterationsPerTask;
                    int counter = 0;
                    while(counter < iter) {
                        //Console.WriteLine($"Iteration {iteration}.");
                        for (int i = 0; i < graphSizes.Length; i++)
                        {
                            for (int j = 0; j < temperatures.Length; j++)
                            {
                                for (int tn = 0; tn < weights_neg.Length; tn++)
                                {
                                    var graph = graphCreator.GenerateGraphWithNegativeWeights(graphSizes[i], temperatures[j], weights_neg[tn]);
                                    var start = DateTime.Now;
                                    var found = solverFW.ShortestPathArrayWithNoNegativeLoops(graph);
                                    var runtime = (DateTime.Now - start).TotalMilliseconds;
                                    lock (resultList)
                                    {
                                        Console.WriteLine($"Adding result for graph size {graphSizes[i]}. Iteration: {counter}. Task: {taskNr}.");
                                        resultList.Add(new FloydWarshallRunData(graphSizes[i], temperatures[j], graph.Edges.Count, runtime, found, true, weights_neg[tn]).ToString());
                                    }
                                }

                                for (int tp = 0; tp < weights_pos.Length; tp++)
                                {
                                    var graph = graphCreator.GenerateGraphOnlyNonNegativeWeights(graphSizes[i], temperatures[j], weights_pos[tp]);

                                    var start = DateTime.Now;
                                    var found = solverFW.ShortestPathArrayWithNoNegativeLoops(graph);
                                    var runtime = (DateTime.Now - start).TotalMilliseconds;
                                    lock (resultList)
                                    {
                                        Console.WriteLine($"Adding result for graph size {graphSizes[i]}. Iteration: {counter}. Task: {taskNr}.");
                                        resultList.Add(new FloydWarshallRunData(graphSizes[i], temperatures[j], graph.Edges.Count, runtime, found, false, weights_neg[tp]).ToString());
                                    }
                                }
                            }
                        }
                        counter++;
                    }
                });
            }

            Task.WaitAll(tasks);
            new FinalDataSaver().SaveToCSV(filepath, title, resultList);
        }
    }
}
