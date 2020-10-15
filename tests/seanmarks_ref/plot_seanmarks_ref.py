import numpy as np
import matplotlib.pyplot as plt

# Plot beta F_N seanmarks
file = "F_N_WHAM.out"
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
file = "F_Ntilde_WHAM.out"
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
fig.savefig("free_energy_profiles.png")

np.save("bins.npy", Nt)
np.save("betaF_Ntilde.npy", betaF)

# Plot beta F_Ntilde biased seanmarks
file = "F_Ntilde_biased.out"
Nt = []
betaF = []
with open(file) as f:
    for line in f:
        if line.strip()[0] != "#":
            vals = line.strip().split()
            Nt.append(float(vals[0]))
            betaF.append([float(val) for val in vals[1:]])

betaF = np.array(betaF).T
print(betaF.shape)

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
for i in range(betaF.shape[0]):
    ax.plot(Nt, betaF[i, :])
ax.set_xlabel(r"Probe waters, $N$")
ax.set_ylabel(r"$\beta F$")
fig.savefig("free_energy_profile_biased.png")
