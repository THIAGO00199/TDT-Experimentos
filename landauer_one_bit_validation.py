import numpy as np
from scipy.integrate import solve_ivp

beta = 0.4
kT = 1.0 / beta
nu = 1.0
eps_f = 15.0
barriers = [18.0, 20.0, 22.0]

def run(B, tau, n_eval=5001):
    # Fixed transition-state energy B; E0=0, E1=epsilon(t).
    # Arrhenius rates satisfy k01/k10 = exp(-beta*epsilon).
    k01 = nu * np.exp(-beta * B)
    tau_relax_0 = 1.0 / (2.0 * k01)
    deps = eps_f / tau

    def rhs(t, y):
        eps = deps * t
        k10 = nu * np.exp(-beta * (B - eps))
        p1 = y[0]
        return [k01 * (1.0 - p1) - k10 * p1]

    ts = np.linspace(0.0, tau, n_eval)
    sol = solve_ivp(rhs, (0.0, tau), [0.5], t_eval=ts,
                    method="DOP853", rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    eps = deps * ts
    p1 = sol.y[0]
    W = np.trapezoid(p1 * deps, ts)
    p_eq_f = 1.0 / (1.0 + np.exp(beta * eps_f))
    F_i = -kT * np.log(2.0)
    F_f = -kT * np.log(1.0 + np.exp(-beta * eps_f))
    dF = F_f - F_i
    return {
        "B": B, "tau_protocol": tau, "tau_relax_initial": tau_relax_0,
        "eta": tau/tau_relax_0, "W_on": W, "DeltaF_exact": dF,
        "Landauer_kTln2": kT*np.log(2.0), "p1_final": float(p1[-1]),
        "p1_equilibrium_final": float(p_eq_f),
        "dissipated_work_W_minus_DeltaF": W-dF,
        "solver_n": int(len(ts))
    }

out = []
for B in barriers:
    k01 = nu*np.exp(-beta*B)
    tau_relax = 1.0/(2*k01)
    out.append(run(B, 0.01*tau_relax))
    out.append(run(B, 100.0*tau_relax))

print("ONE-BIT LANDAUER VALIDATION")
print(f"beta={beta}, kT={kT}, nu={nu}, epsilon_final={eps_f}")
print("Protocol: epsilon(t)=epsilon_final*t/tau; fixed absolute transition-state energy B.")
print("Columns: B, regime, eta, W, DeltaF, W-DeltaF, p1_final")
for i, r in enumerate(out):
    regime = "FAST" if i % 2 == 0 else "SLOW"
    print(f"{r['B']:.1f} {regime:4s} eta={r['eta']:.6g} "
          f"W={r['W_on']:.9f} DeltaF={r['DeltaF_exact']:.9f} "
          f"D={r['dissipated_work_W_minus_DeltaF']:.9f} "
          f"p1f={r['p1_final']:.9g}")
print(f"Landauer asymptote kT ln2 = {kT*np.log(2):.9f}")
print("Detailed balance check (all epsilon): k01/k10 = exp(-beta*epsilon).")
