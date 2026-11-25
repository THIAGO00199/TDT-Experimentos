#!/usr/bin/env python3
import os, json, time
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

OUT="/home/iknowyou/TDT_Experimentos"
NX=128; NT=80; T=.8; DT=T/NT
NU=.05; TAU=.25
x=np.arange(NX)*2*np.pi/NX
kk=np.fft.fftfreq(NX,d=1/NX); ik=1j*kk
mask=(np.abs(kk)<=NX/3).astype(float)
modes=np.arange(1,10)
basis=np.stack([f(m*x) for f in (np.cos,np.sin) for m in modes],axis=0).T
IK=jnp.asarray(ik); MK=jnp.asarray(mask); BJ=jnp.asarray(basis)
def deriv(v): return jnp.fft.ifft(IK*jnp.fft.fft(v)).real
def rhs(y):
 u=y[:NX]; q=y[NX:]
 ux=deriv(u); qx=deriv(q)
 adv=jnp.fft.ifft(jnp.fft.fft(u*ux)*MK).real
 return jnp.concatenate((-adv+qx,(NU*ux-q)/TAU))
def rk4_y(y):
 k1=rhs(y); k2=rhs(y+DT*k1/2); k3=rhs(y+DT*k2/2); k4=rhs(y+DT*k3)
 return y+DT*(k1+2*k2+2*k3+k4)/6
jac=jax.jit(jax.jacfwd(rhs))
rhs_jit=jax.jit(rhs)
def A_at(y): return np.asarray(jac(jnp.asarray(y)))
def main():
 t0=time.time()
 y=np.concatenate((np.sin(x),np.zeros(NX)))
 # B maps 18 actuator coefficients to u forcing field.
 B=np.zeros((2*NX,2*len(modes))); B[:NX,:]=basis
 G=np.zeros((2*NX,2*NX))
 eye=np.eye(2*NX)
 def gdot(yy,GG):
  A=A_at(yy)
  return A@GG+GG@A.T+B@B.T
 eigmins=[]; conds=[]; traces=[]; ranks=[]
 for n in range(NT):
  k1y=np.asarray(rhs_jit(jnp.asarray(y)))
  k1g=gdot(y,G)
  y2=y+DT*k1y/2; G2=G+DT*k1g/2
  k2y=np.asarray(rhs_jit(jnp.asarray(y2))); k2g=gdot(y2,G2)
  y3=y+DT*k2y/2; G3=G+DT*k2g/2
  k3y=np.asarray(rhs_jit(jnp.asarray(y3))); k3g=gdot(y3,G3)
  y4=y+DT*k3y; G4=G+DT*k3g
  k4y=np.asarray(rhs_jit(jnp.asarray(y4))); k4g=gdot(y4,G4)
  y=y+DT*(k1y+2*k2y+2*k3y+k4y)/6
  G=G+DT*(k1g+2*k2g+2*k3g+k4g)/6
  G=(G+G.T)/2
 ev=np.linalg.eigvalsh(G)
 ev=np.maximum(ev,0)
 tol=max(ev.max()*1e-10,1e-14)
 rank=int(np.sum(ev>tol))
 positive=ev[ev>tol]
 result={
  "config":{"NX":NX,"NT":NT,"T":T,"dt":DT,"nu":NU,"tau":TAU,
   "base_initial_u":"sin(x)","base_initial_q":"0",
   "linearization":"full 2*NX-dimensional semidiscrete nonlinear PDE Jacobian along zero-control reference trajectory",
   "actuators":"real-space samples of cos/sin Fourier modes k=1..9",
   "gramian":"integrated dG/dt=A(t)G+G A(t)^T+B B^T using RK4"},
  "state_final_norm":float(np.linalg.norm(y)),
  "gramian_eigenvalues_ascending":ev.tolist(),
  "gramian_max_eigenvalue":float(ev.max()),
  "gramian_min_eigenvalue":float(ev.min()),
  "numerical_rank_tol":float(tol),
  "numerical_rank":rank,
  "state_dimension":2*NX,
  "positive_eigenvalue_condition_number":float(positive.max()/positive.min()) if len(positive) else None,
  "gramian_trace":float(np.trace(G)),
  "elapsed_seconds":time.time()-t0,
  "interpretation":"Rank is local finite-horizon reachability of the time-varying linearization, not nonlinear global controllability and not reachability under pointwise actuator constraints."
 }
 path=os.path.join(OUT,"phase6_full_trajectory_gramian_nx128.json")
 with open(path,"w") as f: json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!="gramian_eigenvalues_ascending"},indent=2))
 print("EIGENVALUES",np.array2string(ev,precision=4,separator=","))
 print("SAVED",path)
if __name__=="__main__": main()
