import os
import numpy as np
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self):
        self.loss = list()
        self.dir = "results"
        self.filename = "ssl_training_curve.png"

    def update(self, entry):
        self.loss.append(entry)

    def plot(self):
        min_loss = np.argmin(self.loss)
        max_loss = np.argmax(self.loss)

        plt.figure(figsize=(8, 6))
        plt.title("SSL Training Loss")
        plt.plot(self.loss, label="Training Loss")
        plt.scatter(min_loss, self.loss[min_loss], c='r')
        plt.text(min_loss, self.loss[min_loss], f"{round(self.loss[min_loss], 4)}", c='k')
        plt.ylim(0, self.loss[max_loss] + 0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        os.makedirs(self.dir, exist_ok=True)
        plt.savefig(os.path.join(self.dir, self.filename))
