import os
import pandas as pd

class Table:
    def __init__(self):
        self.dir = os.path.join('results')
        os.makedirs(self.dir, exist_ok=True)
        self.filename = 'pattern_ranks.tsv'

    def insert(self, map, mrr, p1, p5):
        df = pd.DataFrame({
            'map': [map],
            'mrr': [mrr],
            'p@1': [p1],
            'p@5': [p5]
        })

        df.to_csv(os.path.join(self.dir, self.filename), sep='\t', index=False)
