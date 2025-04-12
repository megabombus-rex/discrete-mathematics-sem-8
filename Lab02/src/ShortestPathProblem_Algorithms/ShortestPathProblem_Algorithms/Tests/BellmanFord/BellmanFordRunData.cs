using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ShortestPathProblem_Algorithms.Tests.BellmanFord
{
    public record BellmanFordRunData(int NodeCount, double Temperature, int FinalEdgeCount, double RuntimeInMS, bool PathWasFound, bool NegativeWeightsIncluded)
    {
        public override string ToString()
        {
            return $"{NodeCount},{Temperature},{FinalEdgeCount},{RuntimeInMS},{PathWasFound},{NegativeWeightsIncluded}";
        }
    }
}
