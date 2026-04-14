import os
import pandas as pd

class Table:
    def __init__(self):
        self.dir = os.path.join('results')
        os.makedirs(self.dir, exist_ok=True)
        self.filename = 'clustering_nmi.tsv'

    def insert(self, nmi, simclr_nmi):
        df = pd.DataFrame({
            'nmi': [nmi],
            'nmi (simclr + lora)': [simclr_nmi],
            'difference (nmi)': [simclr_nmi - nmi]
        })

        df.to_csv(os.path.join(self.dir, self.filename), sep='\t', index=False)
