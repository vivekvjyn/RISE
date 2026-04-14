import os
import pandas as pd

class Table:
    def __init__(self, raga):
        self.dir = os.path.join('results')
        os.makedirs(self.dir, exist_ok=True)
        self.filename = 'classification_f1.tsv'

        if not os.path.exists(os.path.join(self.dir, self.filename)):
            df = pd.DataFrame(columns=[
                'raga',
                'f1',
                'f1 (simclr + lora)',
                'difference'
            ])
            df.to_csv(os.path.join(self.dir, self.filename), sep='\t', index=False)

        self.raga = raga

    def insert(self, f1, simclr_f1):
        df = pd.read_csv(os.path.join(self.dir, self.filename), sep='\t')

        experiment = (df['raga'] == self.raga)

        if experiment.any():
            df.loc[experiment, ['f1', 'f1 (simclr + lora)', 'difference']] = [f1, simclr_f1, simclr_f1 - f1]
        else:
            df.loc[len(df)] = {
                'raga': self.raga,
                'f1': f1,
                'f1 (simclr + lora)': simclr_f1,
                'difference': simclr_f1 - f1
            }

        df.to_csv(os.path.join(self.dir, self.filename), sep='\t', index=False)
