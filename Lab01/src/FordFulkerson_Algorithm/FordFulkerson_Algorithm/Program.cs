using FordFulkerson_Algorithm;

var runner = new FordFulkersonRunner();

var testResult = runner.GenerateGraphAndCalculateFlow(100, 2.5, 30);
runner.ShowResult(testResult);

var testSizes = new int[] {
    10, 100, 1000, 10000, 100000, //1000000,
};

var temperatures = new double[]
{
    0.5, 1.0, 2.5, 3.5 //5.0, 10.0
};

var weight = 30;

for (int i = 0; i < testSizes.Length; i++)
{
    for (int j = 0; j < temperatures.Length; j++)
    {
        var result = runner.GenerateGraphAndCalculateFlow(testSizes[i], temperatures[j], weight);
        runner.ShowResult(result);
    }
}


//int[,] graph = new int[,]
//{
//    { 0, 16, 13, 0, 0, 0 }, 
//    { 0, 0, 10, 12, 0, 0 },
//    { 0, 4, 0, 0, 14, 0 },  
//    { 0, 0, 9, 0, 0, 20 },
//    { 0, 0, 0, 7, 0, 4 },   
//    { 0, 0, 0, 0, 0, 0 }
//};
