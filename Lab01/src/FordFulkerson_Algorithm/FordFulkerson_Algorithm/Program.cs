using BenchmarkDotNet.Running;
using FordFulkerson_Algorithm;
using FordFulkerson_Algorithm.Data;

var runner = new FordFulkersonRunner();

//var testResult = runner.GenerateGraphAndCalculateFlow(100, 2.5, 30);
//runner.ShowResult(testResult);

var testSizes = new int[] {
    10, 100, 1000, 10000, 20000, 30000, 50000//100000, //1000000,
};

var temperatures = new double[]
{
    0.5, 1.0, 2.5, 3.5, 5.0, 10.0
};

var weight = 30;
var graphCreator = new GraphCreator();

//for (int i = 0; i < testSizes.Length; i++)
//{
//    for (int j = 0; j < temperatures.Length; j++)
//    {
//        var result = runner.GenerateGraphAndCalculateFlow(graphSize: testSizes[i], temperature: temperatures[j], weight);
//        runner.ShowResult(result);
//    }
//}

var testCount = 10;
var meanRuntimeSum = 0.0;

Console.WriteLine($"Test done for {testCount} iterations.");

// for edge = 2*nodes
//for (int j = 0; j < testSizes.Length; j++)
//{
//    var runtimes = new double[testCount];
//    for (int i = 0; i < testCount; i++)
//    {
//        var result = runner.GenerateGraphAndCalculateFlow(testSizes[j], 1.0, weight);
//        runner.ShowResult(result);

//        runtimes[i] = result.Duration.TotalMilliseconds;
//        meanRuntimeSum += result.Duration.TotalMilliseconds;
//    }
//    var meanRuntime = meanRuntimeSum / testCount;

//    var stdDev = 0.0;

//    for (int i = 0; i < testCount; i++)
//    {
//        stdDev += Math.Pow(runtimes[i] - meanRuntime, 2);
//    }
//    stdDev = stdDev / testCount;
//    stdDev = Math.Sqrt(stdDev);

//    Console.WriteLine($"Mean runtime for a graph of size {testSizes[j]} and with {testSizes[j] * 2} edges, mean runtime = {meanRuntime} and std deviation = {stdDev}");

//}

// for edge = 3.5*nodes
//for (int j = 0; j < testSizes.Length; j++)
//{
//    var runtimes = new double[testCount];
//    for (int i = 0; i < testCount; i++)
//    {
//        var result = runner.GenerateGraphAndCalculateFlow(testSizes[j], 2.5, weight);
//        runner.ShowResult(result);

//        runtimes[i] = result.Duration.TotalMilliseconds;
//        meanRuntimeSum += result.Duration.TotalMilliseconds;
//    }
//    var meanRuntime = meanRuntimeSum / testCount;

//    var stdDev = 0.0;

//    for (int i = 0; i < testCount; i++)
//    {
//        stdDev += Math.Pow(runtimes[i] - meanRuntime, 2);
//    }
//    stdDev = stdDev / testCount;
//    stdDev = Math.Sqrt(stdDev);

//    Console.WriteLine($"Mean runtime for a graph of size {testSizes[j]} and with {testSizes[j] * 3.5} edges, mean runtime = {meanRuntime} and std deviation = {stdDev}");
//}



//Graph graph = graphCreator.GenerateGraphStructure(25, 2.5, weight);

//var testR = runner.TestAndSaveGraph(graph, 2.5);
//runner.ShowResult(testR);

//var smallBadGraph = new Graph();
//var smallRandomGraph = graphCreator.GenerateGraphStructure(4, 0.0, 1000000);

//var nodeS = new Node(0);
//var node1 = new Node(1);
//var node2 = new Node(2);
//var nodeT = new Node(3);

//var edgeS1 = new Edge() { StartNodeNr = 0, EndNodeNr = 1, Capacity = 1000000 };
//var edgeS2 = new Edge() { StartNodeNr = 0, EndNodeNr = 2, Capacity = 1000000 };
//var edge12 = new Edge() { StartNodeNr = 1, EndNodeNr = 2, Capacity = 1 };
//var edge1T = new Edge() { StartNodeNr = 1, EndNodeNr = 3, Capacity = 1000000 };
//var edge2T = new Edge() { StartNodeNr = 2, EndNodeNr = 3, Capacity = 1000000 };

//nodeS.Edges.Add(edgeS1);
//nodeS.Edges.Add(edgeS2);
//node1.Edges.Add(edge12);
//node1.Edges.Add(edge1T);
//node2.Edges.Add(edge2T);

//smallBadGraph.AddNode(nodeS);
//smallBadGraph.AddNode(node1);
//smallBadGraph.AddNode(node2);
//smallBadGraph.AddNode(nodeT);

//var result = runner.TestAndSaveGraph(smallBadGraph, 0.0);
//var result2 = runner.TestAndSaveGraph(smallRandomGraph, 0.0);
//runner.ShowResult(result);
//runner.ShowResult(result2);

//var graph = new Graph();
//var node0 = new Node(0);
//var node1 = new Node(1);
//var node2 = new Node(2);
//var node3 = new Node(3);
//var node4 = new Node(4);
//var node5 = new Node(5);

//var edge01 = new Edge() { StartNodeNr = 0, EndNodeNr = 1, Capacity = 16 };
//var edge02 = new Edge() { StartNodeNr = 0, EndNodeNr = 2, Capacity = 13 };

//var edge13 = new Edge() { StartNodeNr = 1, EndNodeNr = 3, Capacity = 12 };

//var edge21 = new Edge() { StartNodeNr = 2, EndNodeNr = 1, Capacity = 4 };
//var edge24 = new Edge() { StartNodeNr = 2, EndNodeNr = 4, Capacity = 14 };

//var edge32 = new Edge() { StartNodeNr = 3, EndNodeNr = 2, Capacity = 9 };
//var edge35 = new Edge() { StartNodeNr = 3, EndNodeNr = 5, Capacity = 20 };

//var edge43 = new Edge() { StartNodeNr = 4, EndNodeNr = 3, Capacity = 7 };
//var edge45 = new Edge() { StartNodeNr = 4, EndNodeNr = 5, Capacity = 4 };

//node0.Edges.Add(edge01);
//node0.Edges.Add(edge02);

//node1.Edges.Add(edge13);

//node2.Edges.Add(edge21);
//node2.Edges.Add(edge24);

//node3.Edges.Add(edge32);
//node3.Edges.Add(edge35);

//node4.Edges.Add(edge43);
//node4.Edges.Add(edge45);

//graph.AddNode(node0);
//graph.AddNode(node1);
//graph.AddNode(node2);
//graph.AddNode(node3);
//graph.AddNode(node4);
//graph.AddNode(node5);

//var res = runner.TestAndSaveGraph(graph, 0);
//runner.ShowResult(res);

//var ff = new MaxFlowFordFulkerson();

//var flow = ff.FordFulkerson(graph, 0, 5);

//Console.WriteLine($"Flow: {flow}");

//for (int i = 0; i < testSizes.Length; i++)
//{
//    for (int j = 0; j < temperatures.Length; j++)
//    {
//        var result = runner.GenerateGraphAndCalculateFlow(testSizes[i], temperatures[j], weight);
//        runner.ShowResult(result);
//    }
//}


//int[,] graph = new int[,]
//{
//    { 0, 16, 13, 0, 0, 0 }, 
//    { 0, 0, 10, 12, 0, 0 },
//    { 0, 4, 0, 0, 14, 0 },  
//    { 0, 0, 9, 0, 0, 20 },
//    { 0, 0, 0, 7, 0, 4 },   
//    { 0, 0, 0, 0, 0, 0 }
//};


