#!/usr/bin/env python3
import json, time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import phase6_projected_subspace as m

BASE=Path("/home/iknowyou/TDT_Experimentos")
def H(z):
    u,q=z
    return 0.5*m.dx*jnp.sum(u*u) + (m.TAU/(2*m.NU))*m.dx*jnp.sum(q*q)
def account_rollout(z0, cs):
    def step(z,c):
        f=m.BJ@c
        k1=m.rhs(z,c)
        z2=m.project_state((z[0]+m.DT*k1[0]/2,z[1]+m.DT*k1[1]/2))
        k2=m.rhs(z2,c)
        z3=m.project_state((z[0]+m.DT*k2[0]/2,z[1]+m.DT*k2[1]/2))
        k3=m.rhs(z3,c)
        z4=m.project_state((z[0]+m.DT*k3[0],z[1]+m.DT*k3[1]))
        k4=m.rhs(z4,c)
        zn=m.project_state((z[0]+m.DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
                            z[1]+m.DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6))
        def rates(zz):
            u,q=zz
            w=m.dx*jnp.sum(f*u)
            d=(m.dx/m.NU)*jnp.sum(q*q)
            return w,d
        r1=rates(z); r2=rates(z2); r3=rates(z3); r4=rates(z4)
        w=m.DT*(r1[0]+2*r2[0]+2*r3[0]+r4[0])/6
        d=m.DT*(r1[1]+2*r2[1]+2*r3[1]+r4[1])/6
        return zn,(w,d)
    zf,(ws,ds)=jax.lax.scan(step,z0,cs)
    return zf,jnp.sum(ws),jnp.sum(ds)

def main():
    t0=time.time()
    s0=m.make_sample(20261112)
    start=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
    old=np.load(BASE/"phase6_pareto_controls.npz")["lambda_100"]
    expanded=np.load(BASE/"phase6_pareto_controls_F10_R50.npz")
    cases=[("original_F2_rate10_lambda100",old)]
    for key in ("lambda_10","lambda_30","lambda_100"):
        cases.append(("expanded_F10_rate50_"+key,expanded[key]))
    cases.append(("zero_control_baseline",np.zeros((m.NT,m.NC))))
    out=[]
    for name,cs in cases:
        zf,W,D=account_rollout(start,jnp.asarray(cs))
        H0=float(H(start)); Hf=float(H(zf)); delta=Hf-H0
        W=float(W); D=float(D); residual=W-delta-D
        scale=max(abs(W),abs(delta),abs(D),1e-30)
        out.append({"case":name,"H_initial":H0,"H_final":Hf,"delta_H":delta,
          "W_work":W,"D_dissipation":D,"R_E":residual,
          "relative_residual":residual/scale,
          "endpoint_u_rms_vs_target":float(jnp.sqrt(jnp.mean((zf[0]-m.project_state((jnp.asarray(m.make_sample(20261113)[0]),jnp.asarray(m.make_sample(20261113)[1])))[0])**2))),
          "endpoint_q_rms_vs_target":float(jnp.sqrt(jnp.mean((zf[1]-m.project_state((jnp.asarray(m.make_sample(20261113)[0]),jnp.asarray(m.make_sample(20261113)[1])))[1])**2)))})
        print(json.dumps(out[-1]),flush=True)
    payload={"config":{"NX":m.NX,"NT":m.NT,"T":m.T,"dt":m.DT,"nu":m.NU,"tau":m.TAU,
      "quadrature":"RK4 stage quadrature of instantaneous work integral(f*u) and dissipation integral(q^2/nu), matching the RK4 state integrator",
      "spatial_integral":"periodic Fourier grid, dx=2pi/NX",
      "model":"u_t + u*u_x = q_x + f; q_t=(nu*u_x-q)/tau",
      "energy":"H=1/2 integral(u^2) dx + tau/(2nu) integral(q^2) dx",
      "residual":"R_E=W-delta_H-D",
      "caveat":"finite dt and projected Fourier dynamics; residual includes time-integration and roundoff error; controls are finite-dimensional and local-optimization outputs"},
      "cases":out,"elapsed_seconds":time.time()-t0}
    path=BASE/"phase6_energy_balance_results.json"
    path.write_text(json.dumps(payload,indent=2))
    print("SAVED",path,flush=True)
if __name__=="__main__": main()
