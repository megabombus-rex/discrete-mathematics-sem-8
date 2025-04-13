using BellmanFordAlgorithm;
using GraphData;
using ShortestPathProblem_Algorithms.DataSavers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ShortestPathProblem_Algorithms.Tests.BellmanFord
{
    public class BFSolverTest_2
    {
        public void Run()
        {
            BellmanFordSolver solverBF = new BellmanFordSolver();

            var temperatures = new double[] { 0.5, 1.0, 2.0, 3.0 };

            var graphSizes = new int[] { 10, 100, 500, 1000, 5000 };

            var weights_neg = new int[] { 30, 50, 100 };
            var weights_pos = new int[] { 30, 50, 100 };

            var filepath = $"{Environment.CurrentDirectory}\\..\\..\\..\\Results\\BellmanFord\\BellmanFordTests.csv";
            var title = $"BFSolverTest x100 iterations, 5k V, 20k E max, {DateTime.Now}";
            var resultList = new List<string>(graphSizes.Length * temperatures.Length * (weights_neg.Length + weights_pos.Length));

            // newline - test, date
            // node_count, temperature, final_edge_count, runtime_in_ms, path_was_found, negative_weights_included

            var iterationsLimit = 100;

            var taskCount = 10;
            var iterationsPerTask = iterationsLimit / taskCount;

            var tasks = new Task[taskCount];

            for (int t = 0; t < taskCount; t++)
            {
                var it = t * taskCount;
                var itC = t * iterationsPerTask;
                var task = new Task(() =>
                {
                    var graphCreator = new GraphGenerator.GraphCreator();
                    //Console.WriteLine($"Running task {t}.");
                    for (var iteration = it; iteration < iterationsPerTask; iteration++)
                    {
                        //Console.WriteLine($"Iteration {iteration}.");
                        for (int i = 0; i < graphSizes.Length; i++)
                        {
                            for (int j = 0; j < temperatures.Length; j++)
                            {
                                for (int tn = 0; tn < weights_neg.Length; tn++)
                                {
                                    var graph = graphCreator.GenerateGraphWithNegativeWeights(graphSizes[i], temperatures[j], weights_neg[tn]);
                                    var start = DateTime.Now;
                                    var found = solverBF.ShortestPathPossible(graph, 0);
                                    var runtime = (DateTime.Now - start).TotalMilliseconds;
                                    lock (resultList){
                                        resultList.Add(new BellmanFordRunData(graphSizes[i], temperatures[j], graph.Edges.Count, runtime, found, true).ToString());
                                    }
                                }

                                for (int tp = 0; tp < weights_pos.Length; tp++)
                                {
                                    var graph = graphCreator.GenerateGraphOnlyNonNegativeWeights(graphSizes[i], temperatures[j], weights_pos[tp]);

                                    var start = DateTime.Now;
                                    var found = solverBF.ShortestPathPossible(graph, 0);
                                    var runtime = (DateTime.Now - start).TotalMilliseconds;
                                    lock (resultList)
                                    {
                                        resultList.Add(new BellmanFordRunData(graphSizes[i], temperatures[j], graph.Edges.Count, runtime, found, false).ToString());
                                    }
                                }
                            }
                        }
                    }

                });

                task.Start();
                tasks[t] = task;
            }

            Task.WaitAll(tasks);
            new BellmanFordSaver().SaveToCSV(filepath, title, resultList);
        }
    }
}
