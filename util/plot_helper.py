# Python file to plot important quanities
import matplotlib.pyplot as plt
import numpy as np

def Shannon_Entropy(batches, entropy, save_path):
  entropy = np.array(entropy)
  batches = np.array(batches)

  # Compute running average
  running_avg = np.cumsum(entropy) / np.arange(1, len(entropy)+1)

  plt.figure(figsize=(8,5))
  plt.plot(batches, entropy, marker='o', color='blue', label='Entropy')
  plt.plot(batches, running_avg, marker='x', color='red', linestyle='--', label='Running Average')
  plt.xlabel("Batch Number")
  plt.ylabel("Shannon Entropy")
  plt.title("Source Convergence (Shannon Entropy)")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(save_path, dpi=300)
  plt.close()


def K_Effective(batches, keff, save_path="keff_plot.png"):
  batches = np.array(batches)
  keff = np.array(keff)

  # Compute running average
  running_avg = np.cumsum(keff) / np.arange(1, len(keff)+1)

  plt.figure(figsize=(8,5))

  plt.plot(batches, keff, marker='o', color='blue', label='k-effective')

  plt.plot(batches, running_avg, marker='x', color='red', linestyle='--', label='Running Average')
  plt.xlabel("Batch Number")
  plt.ylabel(r"$k_{eff}$")
  plt.title("k-effective Convergence")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(save_path, dpi=300)
  plt.close()
