import numpy as np
import pandas as pd

__all__ = ["FakeDendrogram", "FakeDendrogramNode"]


class FakeDendrogramNode:
    def __init__(self, num, index):
        self.num = num
        self.descr = f"Cluster #{num}"
        np.random.seed(self.num)
        self.abs_freqs = pd.Series(np.random.randint(low=0, high=200, size=len(index)), index=index)
        self.rel_freqs = self.abs_freqs / sum(self.abs_freqs) * 100.0

    def __repr__(self):
        return f"<{self.descr}, PC freqs: {list(self.abs_freqs)}>"

    @property
    def distribution(self):
        return self.rel_freqs


class FakeDendrogram:
    def __init__(self, index, num_nodes_below_p_cutoff):
        self.num_nodes_below_p_cutoff = num_nodes_below_p_cutoff
        self.nodes_below_p_cutoff = [FakeDendrogramNode(i, index) for i in range(self.num_nodes_below_p_cutoff)]
