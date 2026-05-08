import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("safety_utility_vaccine_lisa_results.csv")

df["Label"] = [
    "Baseline",
    "LISA\n2e-5lr, 3ep, 1%",
    "LISA\n5e-5lr, 3ep, 5%",
    "LISA\n5e-5lr, 5ep, 10%",
    "Vaccine\n2e-5lr, 3ep, 1%",
    "Vaccine\n5e-5lr, 3ep, 5%",
    "Vaccine\n5e-5lr, 5ep, 10%",
]

x = np.arange(len(df))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: ASR
ax1.bar(x, df["ASR"] * 100)
ax1.set_ylabel("ASR (%)")
ax1.set_title("Attack Success Rate Across Models")
ax1.set_ylim(0, max(df["ASR"] * 100) + 1)

# Bottom: SST-2 Accuracy
ax2.bar(x, df["SST2_Accuracy"] * 100)
ax2.set_ylabel("SST-2 Accuracy (%)")
ax2.set_title("Utility Across Models")
ax2.set_ylim(0, 100)

ax2.set_xticks(x)
ax2.set_xticklabels(df["Label"], rotation=30, ha="right")

plt.tight_layout()
plt.savefig("figures/defense_safety_utility_two_plots.png", dpi=300, bbox_inches="tight")
plt.show()

