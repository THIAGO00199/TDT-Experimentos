#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import phase6_projected_subspace as m

OUT=Path("/home/iknowyou/TDT_Experimentos/phase6_pareto_results.json")
CTRL=Path("/home/iknowyou/TDT_Experimentos/phase6_pareto_controls.npz")
LAMBDAS=[0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0,3.0,10.0,30.0,100.0]
MAXITER=800

def main():
 t0=time.time()
 s0=m.make_sample(20261112); s1=m.make_sample(20261113)
 start=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
 target=m.project_state((jnp.asarray(s1[0]),jnp.asarray(s1[1])))
 su=max(float(jnp.sqrt(jnp.mean(target[0]**2))),.1)
 sq=max(float(jnp.sqrt(jnp.mean(target[1]**2))),.02)
 def metrics(raw):
  cs=m.project(jnp.asarray(raw.reshape(m.NT,m.NC)))
  h=m.rollout(start,cs)
  eu=float(jnp.sqrt(jnp.mean((h[0][-1]-target[0])**2)))
  eq=float(jnp.sqrt(jnp.mean((h[1][-1]-target[1])**2)))
  spatial=np.asarray(cs)@m.basis.T
  return cs,eu,eq,float(np.max(np.abs(spatial))),float(np.max(np.abs(np.diff(spatial,axis=0)))/m.DT)
 def objective(flat,lam):
  raw=flat.reshape(m.NT,m.NC)
  cs=m.project(raw); h=m.rollout(start,cs)
  eu=jnp.mean(((h[0][-1]-target[0])/su)**2)
  eq=jnp.mean(((h[1][-1]-target[1])/sq)**2)
  effort=jnp.mean(cs**2)
  slew=jnp.mean(jnp.diff(cs,axis=0)**2)/(m.DT**2)
  return eu+lam*eq+1e-5*effort+1e-7*slew
 # Obtain the best u-only seed-2 solution as requested, with a fresh reproducible solve.
 rng=np.random.default_rng(2)
 raw=rng.normal(0,.15,(m.NT,m.NC))
 ufun=jax.jit(lambda v:objective(v,0.0))
 ugrad=jax.jit(jax.grad(lambda v:objective(v,0.0)))
 opt=minimize(lambda v:float(ufun(jnp.asarray(v))),raw.ravel(),
  jac=lambda v:np.asarray(ugrad(jnp.asarray(v)),float),method="L-BFGS-B",
  bounds=[(-20,20)]*(m.NT*m.NC),
  options={"maxiter":MAXITER,"ftol":1e-12,"gtol":1e-8,"maxls":40,"maxcor":20})
 raw=opt.x.reshape(m.NT,m.NC)
 cs,eu,eq,mf,mr=metrics(raw)
 print(json.dumps({"phase":"u_seed","success":bool(opt.success),"iterations":int(opt.nit),
  "u_rms":eu,"q_rms":eq,"max_force":mf,"max_rate":mr}),flush=True)
 rows=[]; controls={}
 # Include exact u-only point, then warm-start each increasing q penalty.
 for i,lam in enumerate(LAMBDAS):
  fun=jax.jit(lambda v:objective(v,lam))
  grad=jax.jit(jax.grad(lambda v:objective(v,lam)))
  opt=minimize(lambda v:float(fun(jnp.asarray(v))),raw.ravel(),
   jac=lambda v:np.asarray(grad(jnp.asarray(v)),float),method="L-BFGS-B",
   bounds=[(-20,20)]*(m.NT*m.NC),
   options={"maxiter":MAXITER,"ftol":1e-12,"gtol":1e-8,"maxls":40,"maxcor":20})
  raw=opt.x.reshape(m.NT,m.NC)
  cs,eu,eq,mf,mr=metrics(raw)
  row={"lambda_q":lam,"success":bool(opt.success),"message":str(opt.message),
   "iterations":int(opt.nit),"objective":float(opt.fun),
   "endpoint_u_rms":eu,"endpoint_q_rms":eq,
   "normalized_u_rmse":eu/su,"normalized_q_rmse":eq/sq,
   "max_abs_f_space":mf,"max_abs_dfdt_space":mr}
  rows.append(row); controls[f"lambda_{i}"]=np.asarray(cs)
  print(json.dumps({"pareto":row}),flush=True)
 np.savez_compressed(CTRL,**controls)
 payload={"config":{"NX":m.NX,"NT":m.NT,"T":m.T,"dt":m.DT,"nu":m.NU,"tau":m.TAU,
  "force_bound_pointwise":m.FMAX,"rate_bound_pointwise":m.RATE_MAX,
  "lambda_q_values":LAMBDAS,"objective":"normalized mean-square u endpoint error + lambda_q * normalized mean-square q endpoint error + small actuator effort/slew regularizers",
  "normalizers":{"u_rms_target_floor":su,"q_rms_target_floor":sq},
  "initialization":"fresh reproduction of seed-2 u-only optimum; each next lambda warm-started from previous lambda",
  "limitations":"finite-dimensional NX32/NT80 projected model; soft endpoint objective; local L-BFGS-B solutions; Pareto points are not certified global optima"},
  "u_seed":{"success":bool(opt.success),"iterations":int(opt.nit)},
  "baseline_no_control":{"u_rms":float(jnp.sqrt(jnp.mean((start[0]-target[0])**2))),
   "q_rms":float(jnp.sqrt(jnp.mean((start[1]-target[1])**2)))},
  "points":rows,"control_archive":str(CTRL),"elapsed_seconds":time.time()-t0}
 OUT.write_text(json.dumps(payload,indent=2))
 print("SAVED",OUT,flush=True); print("CONTROLS",CTRL,flush=True)
 print("ELAPSED",payload["elapsed_seconds"],flush=True)
if __name__=="__main__":main()
