using BellmanFordAlgorithm;
using ShortestPathProblem_Algorithms.Tests.BellmanFord;
using ShortestPathProblem_Algorithms.Tests.FloydWarshall;
using System.Globalization;

BellmanFordSolver solverBF = new BellmanFordSolver();

var ci = new CultureInfo("en-US");
Thread.CurrentThread.CurrentCulture = ci;
Thread.CurrentThread.CurrentUICulture = ci;
//var test = new BFSolverTest_2();
var test = new FWSolverTest_1();

test.Run();