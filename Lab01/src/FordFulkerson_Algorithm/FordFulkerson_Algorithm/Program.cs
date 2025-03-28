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

for (int i = 0; i < testSizes.Length; i++)
{
    for (int j = 0; j < temperatures.Length; j++)
    {
        var result = runner.GenerateGraphAndCalculateFlow(graphSize: testSizes[i], temperature: temperatures[j], weight);
        runner.ShowResult(result);
    }
}


//var graphCreator = new GraphCreator();

//Graph graph = graphCreator.GenerateGraphStructure(10, 2.5, weight);

//var graph = new Graph();
//var node0 = new Node(0);
//var node1 = new Node(1);
//var node2 = new Node(2);
//var node3 = new Node(3);
//var node4 = new Node(4);
//var node5 = new Node(5);

//var edge01 = new Edge() { StartNode = 0, EndNode = 1, Capacity = 16 };
//var edge02 = new Edge() { StartNode = 0, EndNode = 2, Capacity = 13 };

//var edge12 = new Edge() { StartNode = 1, EndNode = 2, Capacity = 10 };
//var edge13 = new Edge() { StartNode = 1, EndNode = 3, Capacity = 12 };

//var edge21 = new Edge() { StartNode = 2, EndNode = 1, Capacity = 4 };
//var edge24 = new Edge() { StartNode = 2, EndNode = 4, Capacity = 14 };

//var edge32 = new Edge() { StartNode = 3, EndNode = 2, Capacity = 9 };
//var edge35 = new Edge() { StartNode = 3, EndNode = 5, Capacity = 20 };

//var edge43 = new Edge() { StartNode = 4, EndNode = 3, Capacity = 7 };
//var edge45 = new Edge() { StartNode = 4, EndNode = 5, Capacity = 4 };

//node0.Edges.Add(edge01);
//node0.Edges.Add(edge02);

//node1.Edges.Add(edge12);
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


