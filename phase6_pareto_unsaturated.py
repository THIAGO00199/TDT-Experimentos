#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import phase6_projected_subspace as m

BASE=Path("/home/iknowyou/TDT_Experimentos")
RESULT=json.loads((BASE/"phase6_pareto_results.json").read_text())
ARCH=np.load(BASE/"phase6_pareto_controls.npz")
# Reconstruct the exact start and target used in the Pareto experiment.
s0=m.make_sample(20261112); s1=m.make_sample(20261113)
start=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
target=m.project_state((jnp.asarray(s1[0]),jnp.asarray(s1[1])))
start_np=tuple(np.asarray(v) for v in start)
target_np=tuple(np.asarray(v) for v in target)

def modal_report(cs,label):
 h=m.rollout(start,jnp.asarray(cs))
 qf=np.asarray(h[1][-1]); qt=target_np[1]; err=qf-qt
 fq=np.fft.rfft(qt)/m.NX
 fe=np.fft.rfft(err)/m.NX
 rows=[]
 # Pair contribution to spatial RMS: sqrt(2)*|Fourier coefficient| for k>0.
 for k in range(1,11):
  target_rms=float(np.sqrt(2)*abs(fq[k]))
  err_rms=float(np.sqrt(2)*abs(fe[k]))
  rows.append({"k":k,"target_mode_rms":target_rms,"residual_mode_rms":err_rms,
   "relative_mode_error":err_rms/(target_rms+1e-30)})
 total=float(np.sqrt(np.mean(err**2)))
 return {"label":label,"q_endpoint_rms":total,"modes":rows}

def main():
 t0=time.time()
 # Existing bounded Pareto endpoint at lambda=100.
 cs_old=ARCH["lambda_100"]
 old=modal_report(cs_old,"original_bounds_F2_rate10_lambda100")
 print(json.dumps({"modal_decomposition":old}),flush=True)

 # Rebuild optimization objective with expanded spatial bounds.
 m.FMAX=10.0
 m.RATE_MAX=50.0
 su=max(float(jnp.sqrt(jnp.mean(target[0]**2))),.1)
 sq=max(float(jnp.sqrt(jnp.mean(target[1]**2))),.02)
 def obj(flat,lam):
  raw=flat.reshape(m.NT,m.NC)
  cs=m.project(raw); h=m.rollout(start,cs)
  eu=jnp.mean(((h[0][-1]-target[0])/su)**2)
  eq=jnp.mean(((h[1][-1]-target[1])/sq)**2)
  effort=jnp.mean(cs**2)
  slew=jnp.mean(jnp.diff(cs,axis=0)**2)/(m.DT**2)
  return eu+lam*eq+1e-5*effort+1e-7*slew
 raw=np.asarray(cs_old).copy()
 lambdas=[10.0,30.0,100.0]
 rows=[]; expanded_controls={}
 for i,lam in enumerate(lambdas):
  fun=jax.jit(lambda v:obj(v,lam))
  grad=jax.jit(jax.grad(lambda v:obj(v,lam)))
  opt=minimize(lambda v:float(fun(jnp.asarray(v))),raw.ravel(),
   jac=lambda v:np.asarray(grad(jnp.asarray(v)),float),
   method="L-BFGS-B",bounds=[(-30,30)]*(m.NT*m.NC),
   options={"maxiter":1200,"ftol":1e-12,"gtol":1e-8,"maxls":50,"maxcor":30})
  raw=opt.x.reshape(m.NT,m.NC)
  cs=np.asarray(m.project(jnp.asarray(raw)))
  h=m.rollout(start,jnp.asarray(cs))
  eu=float(jnp.sqrt(jnp.mean((np.asarray(h[0][-1])-target_np[0])**2)))
  eq=float(jnp.sqrt(jnp.mean((np.asarray(h[1][-1])-target_np[1])**2)))
  spatial=cs@m.basis.T
  maxf=float(np.max(np.abs(spatial)))
  maxrate=float(np.max(np.abs(np.diff(spatial,axis=0)))/m.DT)
  row={"lambda_q":lam,"success":bool(opt.success),"message":str(opt.message),
   "iterations":int(opt.nit),"objective":float(opt.fun),"endpoint_u_rms":eu,
   "endpoint_q_rms":eq,"max_abs_force_space":maxf,"max_abs_rate_space":maxrate,
   "amplitude_bound_active":bool(maxf>9.999),"rate_bound_active":bool(maxrate>49.99)}
  rows.append(row); expanded_controls[f"lambda_{int(lam)}"]=cs
  print(json.dumps({"expanded_bounds":row}),flush=True)
  if lam==100.0:
   print(json.dumps({"expanded_modal_decomposition":modal_report(cs,"expanded_bounds_F10_rate50_lambda100")}),flush=True)
 np.savez_compressed(BASE/"phase6_pareto_controls_F10_R50.npz",**expanded_controls)
 payload={"config":{"Fmax_spatial":10.0,"rate_max_spatial":50.0,"lambda_q_values":lambdas,
  "warm_start":"saved original-bound lambda=100 control; continuation 10 -> 30 -> 100",
  "same_NX_NT_T_nu_tau_and_endpoint_pair":True,
  "objective":"normalized MSE_u + lambda_q normalized MSE_q + original effort/slew regularizers",
  "limitations":"local optimization; finite-dimensional projected model; not proof of topology"},
  "original_bounds_lambda100":old,"expanded_runs":rows,
  "controls_file":str(BASE/"phase6_pareto_controls_F10_R50.npz"),
  "elapsed_seconds":time.time()-t0}
 (BASE/"phase6_pareto_unsaturated_results.json").write_text(json.dumps(payload,indent=2))
 print("SAVED",BASE/"phase6_pareto_unsaturated_results.json",flush=True)
 print("ELAPSED",payload["elapsed_seconds"],flush=True)
if __name__=="__main__":main()
