import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import phase6_projected_subspace as m
base=Path("/home/iknowyou/TDT_Experimentos")
s0=m.make_sample(20261112); s1=m.make_sample(20261113)
start=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
target=m.project_state((jnp.asarray(s1[0]),jnp.asarray(s1[1])))
d=np.load(base/"phase6_pareto_controls.npz")
rows=[]
for lam in [0.0,.001,.003,.01,.03,.1,.3,1.,3.,10.,30.,100.]:
    cs=jnp.asarray(d[f"lambda_{lam:g}"])
    hist=m.rollout(start,cs)
    u=np.asarray(hist[0]); q=np.asarray(hist[1])
    qend=q[-1]; du=np.asarray(u[-1])-np.asarray(target[0]); dq=qend-np.asarray(target[1])
    qhat=np.fft.fft(dq)/m.NX
    modes={}
    for k in range(1,11):
        modes[str(k)]=float(np.sqrt(2)*abs(qhat[k]))
    # Discrete diagnostic energy accounting using midpoint state/control.
    us=.5*(np.vstack([np.asarray(start[0])[None,:],u[:-1]])+u)
    qs=.5*(np.vstack([np.asarray(start[1])[None,:],q[:-1]])+q)
    fs=np.asarray(cs)@m.basis.T
    work=float(m.DT*m.dx*np.sum(us*fs))
    diss=float(m.DT*m.dx*np.sum(qs*qs)/m.NU)
    def H(uu,qq):
        return .5*m.dx*np.sum(uu*uu)+m.TAU/(2*m.NU)*m.dx*np.sum(qq*qq)
    deltaH=H(u[-1],q[-1])-H(np.asarray(start[0]),np.asarray(start[1]))
    rows.append({"lambda_q":lam,"q_error_mode_amplitudes":modes,
       "q_mode4_residual_amplitude":modes["4"],"work_input_midpoint":work,
       "dissipation_midpoint":diss,"delta_H":float(deltaH),
       "balance_residual_W_minus_dH_minus_D":work-deltaH-diss})
out={"note":"q-mode residual amplitudes are Fourier coefficients of endpoint error q(T)-q_target; energy terms are midpoint diagnostics, not a separately converged energy-balance certification.",
     "rows":rows}
(base/"phase6_pareto_mode_energy_diagnostics.json").write_text(json.dumps(out,indent=2))
for r in rows:
 print(json.dumps({"lambda":r["lambda_q"],"q_mode4_residual":r["q_mode4_residual_amplitude"],"W":r["work_input_midpoint"],"D":r["dissipation_midpoint"],"dH":r["delta_H"],"balance_residual":r["balance_residual_W_minus_dH_minus_D"]}))
print("SAVED",base/"phase6_pareto_mode_energy_diagnostics.json")
