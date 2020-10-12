import numpy as np
import matplotlib.pyplot as plt

# Plot beta F_N seanmarks
file = "seanmarks_ref/F_N_WHAM.out"
N = []
betaF = []
betaFerr = []
with open(file) as f:
    for line in f:
        if line[0] != "#":
            vals = line.strip().split()
            N.append(float(vals[0]))
            betaF.append(float(vals[1]))
            betaFerr.append(float(vals[3]))

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax.errorbar(N, betaF, yerr=betaFerr, capsize=3, label="$N_v$")

# Plot beta F_Ntilde seanmarks
file = "seanmarks_ref/F_Ntilde_WHAM.out"
Nt = []
betaF = []
betaFerr = []
with open(file) as f:
    for line in f:
        if line.strip()[0] != "#":
            vals = line.strip().split()
            Nt.append(float(vals[0]))
            betaF.append(float(vals[1]))
            betaFerr.append(float(vals[3]))

ax.errorbar(Nt, betaF, yerr=betaFerr, capsize=3, label=r"$\tilde{N}_v$")
ax.set_xlabel(r"Probe waters, $N$")
ax.set_ylabel(r"$\beta F$")
ax.legend()
fig.savefig("seanmarks_free_energy_profiles.png")
