namespace ShortestPathProblem_Algorithms.Tests.FloydWarshall
{
    public record FloydWarshallRunData(int NodeCount, double Temperature, int FinalEdgeCount, double RuntimeInMS, bool PathWasFound, bool NegativeWeightsIncluded, int Weight)
    {
        public override string ToString()
        {
            return $"{NodeCount},{Temperature},{FinalEdgeCount},{RuntimeInMS},{PathWasFound},{NegativeWeightsIncluded},{Weight}";
        }
    }
}
