"""Generate figures for the scientific report from spatiotemporal field data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/data/user/conversations/conv-d63ebdcf-6bdc-4bf7-a59f-78da38ea0a88/project"
DATA = f"{BASE}/data/spatiotemporal_fields.npz"
FIGDIR = f"{BASE}/output/figures"
DPI = 150

d = np.load(DATA)
t = d["synthetic_time"]
x = d["synthetic_space"]
U = d["synthetic_field"]  # (time, space)

dt = np.diff(t)[0]
dx = np.diff(x)[0]


def second_derivative(u, h, axis):
    """Central-difference second derivative along `axis`, matching the core
    analysis residual (output/analysis.py::compute_residual_field). Boundaries
    are left zero so the spatial-boundary residual is exactly 0 (U=0 at x=+-5).
    Returns an array the same shape as u."""
    out = np.zeros_like(u, dtype=np.float64)
    if axis == 0:
        out[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / h**2
    else:
        out[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / h**2
    return out


def save(fig, path, title=""):
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: contour plot of the synthetic field (time vs space)
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
cf = ax1.contourf(x, t, U, levels=40, cmap="viridis")
fig1.colorbar(cf, ax=ax1, label="Field value")
ax1.set_xlabel("Space")
ax1.set_ylabel("Time")
ax1.set_title("Fig. 1 — Synthetic spatiotemporal field")
save(fig1, f"{FIGDIR}/fig1_synthetic_field.png")

# ---------------------------------------------------------------------------
# Fig 2: residual U_tt - U_xx (central-difference second derivatives)
# ---------------------------------------------------------------------------
U_tt = second_derivative(U, dt, axis=0)
U_xx = second_derivative(U, dx, axis=1)
residual = U_tt - U_xx

fig2, ax2 = plt.subplots(figsize=(8, 5))
cf2 = ax2.contourf(x, t, residual, levels=40, cmap="RdBu_r")
fig2.colorbar(cf2, ax=ax2, label=r"$U_{tt} - U_{xx}$")
ax2.set_xlabel("Space")
ax2.set_ylabel("Time")
ax2.set_title("Fig. 2 — Residual $U_{tt} - U_{xx}$ (central differences)")
save(fig2, f"{FIGDIR}/fig2_residual.png")

# ---------------------------------------------------------------------------
# Fig 3: amplitudes of the 5 dominant spatial modes vs time (FFT, no DC)
# ---------------------------------------------------------------------------
F = np.fft.fft(U, axis=1)          # (time, mode)
amps = np.abs(F)
total = amps.sum(axis=0)
total[0] = 0.0                     # exclude DC mode
kdom = np.argsort(total)[-5:][::-1]
labels = [f"mode {k}" for k in kdom]

fig3, ax3 = plt.subplots(figsize=(7, 5))
for k, lab in zip(kdom, labels):
    ax3.plot(t, amps[:, k], label=lab)
ax3.set_xlabel("Time")
ax3.set_ylabel(r"Amplitude $|\tilde{U}(k,t)|$")
ax3.set_title("Fig. 3 — Amplitude of the 5 dominant spatial modes vs time")
ax3.legend()
save(fig3, f"{FIGDIR}/fig3_modes.png")

# ---------------------------------------------------------------------------
# Fig 4: experimental data, 5 subplots (one per time), 3 channels vs coordinate
# ---------------------------------------------------------------------------
exp_t = d["experimental_time"]
exp_x = d["experimental_space"]
exp_f = d["experimental_field"]   # (time, space, 3)

fig4, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.ravel()
for i, ax in enumerate(axes[:5]):
    profile = exp_f[i]           # (38, 3)
    for c in range(3):
        ax.plot(exp_x, profile[:, c], label=f"channel {c + 1}")
    ax.set_xlabel("Source assay coordinate")
    ax.set_ylabel("Signal")
    ax.set_title(f"t = {exp_t[i]} h")
    ax.legend(fontsize=8)
if len(axes) > 5:
    axes[5].axis("off")
fig4.suptitle("Fig. 4 — Experimental channel profiles at each time point")
save(fig4, f"{FIGDIR}/fig4_experimental.png")

print("All figures written to", FIGDIR)
