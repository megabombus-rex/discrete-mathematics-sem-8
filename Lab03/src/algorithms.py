class MaxCut:
    def __call__(self, graph):
        raise NotImplementedError("MaxCut is an abstract class. Use child class.")
    
class BruteForceMaxCut(MaxCut):
    def __call__(self, graph):
        return []