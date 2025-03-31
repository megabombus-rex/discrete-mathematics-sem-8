using BenchmarkDotNet.Running;
using FordFulkerson_Algorithm.Benchmarks;

var summary = BenchmarkRunner.Run<FordFulkersonBenchmark>();