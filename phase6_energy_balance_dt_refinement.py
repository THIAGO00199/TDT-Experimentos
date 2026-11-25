#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np, jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import phase6_projected_subspace as m
BASE=Path("/home/iknowyou/TDT_Experimentos")
def H(z):
 u,q=z
 return .5*m.dx*jnp.sum(u*u)+(m.TAU/(2*m.NU))*m.dx*jnp.sum(q*q)
def run(z0,cs):
 def step(z,c):
  f=m.BJ@c
  k1=m.rhs(z,c)
  z2=m.project_state((z[0]+m.DT*k1[0]/2,z[1]+m.DT*k1[1]/2)); k2=m.rhs(z2,c)
  z3=m.project_state((z[0]+m.DT*k2[0]/2,z[1]+m.DT*k2[1]/2)); k3=m.rhs(z3,c)
  z4=m.project_state((z[0]+m.DT*k3[0],z[1]+m.DT*k3[1])); k4=m.rhs(z4,c)
  zn=m.project_state((z[0]+m.DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
                      z[1]+m.DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6))
  def rate(zz):
   return m.dx*jnp.sum(f*zz[0]),m.dx/m.NU*jnp.sum(zz[1]**2)
  a,b,c1,d=rate(z),rate(z2),rate(z3),rate(z4)
  w=m.DT*(a[0]+2*b[0]+2*c1[0]+d[0])/6
  diss=m.DT*(a[1]+2*b[1]+2*c1[1]+d[1])/6
  return zn,(w,diss)
 zf,(ww,dd)=jax.lax.scan(step,z0,cs)
 return zf,jnp.sum(ww),jnp.sum(dd)
def main():
 # Recreate the source using the original 80-step discretization first.
 s0=m.make_sample(20261112)
 z0=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
 old=np.load(BASE/"phase6_pareto_controls.npz")["lambda_100"]
 ex=np.load(BASE/"phase6_pareto_controls_F10_R50.npz")
 cases=[("original_F2_rate10_lambda100",old),("expanded_F10_rate50_lambda100",ex["lambda_100"])]
 # Refine time integration while preserving the same piecewise-constant control history.
 m.NT=160; m.DT=m.T/m.NT
 rows=[]
 for name,c in cases:
  cs=np.repeat(c,2,axis=0)
  zf,W,D=run(z0,jnp.asarray(cs))
  dh=float(H(zf)-H(z0)); w=float(W); d=float(D); r=w-dh-d
  rows.append({"case":name,"dt":m.DT,"delta_H":dh,"W_work":w,"D_dissipation":d,
               "R_E":r,"relative_residual":r/max(abs(w),abs(dh),abs(d),1e-30)})
  print(json.dumps(rows[-1]),flush=True)
 out={"validation":"same piecewise-constant controls, same initial state, RK4 dt halved from 0.01 to 0.005",
      "cases":rows}
 p=BASE/"phase6_energy_balance_dt_refinement.json"; p.write_text(json.dumps(out,indent=2))
 print("SAVED",p)
if __name__=="__main__":main()
