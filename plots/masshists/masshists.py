import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path as pa

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
from utils.paths import PROJDIR
import utils.graham23_tables as g23

plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")

################################################################################

# mbins = np.linspace(0, 200, 50)

for col in g23.DF_GW.columns:
    if ("mass" in col) and (not "lower" in col) and (not "upper" in col):
        x = g23.DF_GW[col]
        if np.all(np.isnan(x)):
            continue
        plt.hist(
            x,
            # bins=mbins,
            histtype="step",
            label=col,
            lw=5,
            alpha=0.7,
        )
plt.xlabel(r"Mass ($M_{\odot}$)")
plt.ylabel("N")
plt.legend()
plt.tight_layout()
plt.savefig(__file__.replace(".py", ".png"))
plt.savefig(__file__.replace(".py", ".pdf"))
plt.close()
