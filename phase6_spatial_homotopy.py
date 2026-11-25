#!/usr/bin/env python3
import os,json,time
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np, jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from scipy.optimize import minimize
NX=32; NT=80; T=.8; DT=T/NT; NU=.05; TAU=.25
FMAX=2.; RATE_MAX=10.; MODES=np.arange(1,10); NC=18
x=np.arange(NX)*2*np.pi/NX; dx=2*np.pi/NX
kk=np.fft.fftfreq(NX,d=1/NX); IK=jnp.asarray(1j*kk); MK=jnp.asarray((np.abs(kk)<=NX/3).astype(float))
basis=np.stack([f(m*x) for f in (np.cos,np.sin) for m in MODES],axis=0).T
BJ=jnp.asarray(basis)
def deriv(v): return jnp.fft.ifft(IK*jnp.fft.fft(v)).real
def rhs(z,c):
 u,q=z; ux=deriv(u); qx=deriv(q)
 adv=jnp.fft.ifft(jnp.fft.fft(u*ux)*MK).real
 return (-adv+qx+BJ@c,(NU*ux-q)/TAU)
def rk4(z,c):
 k1=rhs(z,c); k2=rhs((z[0]+DT*k1[0]/2,z[1]+DT*k1[1]/2),c)
 k3=rhs((z[0]+DT*k2[0]/2,z[1]+DT*k2[1]/2),c)
 k4=rhs((z[0]+DT*k3[0],z[1]+DT*k3[1]),c)
 return (z[0]+DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
         z[1]+DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)
def project(raw):
 def step(prev,des):
  fdes=BJ@des; peakf=jnp.max(jnp.abs(fdes))
  des=des*jnp.minimum(1.,FMAX/(peakf+1e-30))
  delta=des-prev; field=BJ@delta
  peak=jnp.max(jnp.abs(field))
  delta=delta*jnp.minimum(1.,RATE_MAX*DT/(peak+1e-30))
  nxt=prev+delta
  return nxt,nxt
 _,cs=jax.lax.scan(step,jnp.zeros(NC),raw)
 return cs
def rollout(z0,cs):
 def step(z,c):
  zn=rk4(z,c); return zn,zn
 _,hist=jax.lax.scan(step,z0,cs)
 return hist
def forward(z):
 for _ in range(NT): z=rk4(z,jnp.zeros(NC))
 return z
def make_sample(seed):
 rng=np.random.default_rng(seed); Z=rng.standard_normal(24)
 u=np.sin(x).copy(); q=np.zeros(NX)
 for j,m in enumerate(range(4,10)):
  u+=.0015*(Z[2*j]*np.cos(m*x)+Z[2*j+1]*np.sin(m*x))
  q+=.0008*(Z[12+2*j]*np.cos(m*x)+Z[13+2*j]*np.sin(m*x))
 return u,q,Z
def main():
 t0=time.time()
 s0=make_sample(20261112); s1=make_sample(20261113)
 start=forward((jnp.asarray(s0[0]),jnp.asarray(s0[1])))
 target=(jnp.asarray(s1[0]),jnp.asarray(s1[1]))
 su=max(float(jnp.sqrt(jnp.mean(target[0]**2))),.1)
 sq=max(float(jnp.sqrt(jnp.mean(target[1]**2))),.02)
 def factory(weight):
  def obj(flat):
   raw=flat.reshape(NT,NC); cs=project(raw); h=rollout(start,cs)
   eu=jnp.mean(((h[0][-1]-target[0])/su)**2)
   eq=jnp.mean(((h[1][-1]-target[1])/sq)**2)
   effort=jnp.mean(cs**2)
   slew=jnp.mean(jnp.diff(cs,axis=0)**2)/(DT**2)
   return weight*(eu+eq)+1e-5*effort+1e-7*slew
  return jax.jit(obj),jax.jit(jax.grad(obj))
 raw=np.zeros((NT,NC)); stages=[]
 for weight in [1e2,1e3,1e4,1e5]:
  fun,grad=factory(weight)
  opt=minimize(lambda v:float(fun(jnp.asarray(v))),
   raw.ravel(),jac=lambda v:np.asarray(grad(jnp.asarray(v)),float),
   method="L-BFGS-B",bounds=[(-20,20)]*(NT*NC),
   options={"maxiter":180,"ftol":1e-10,"gtol":1e-7,"maxls":25,"maxcor":10})
  raw=opt.x.reshape(NT,NC)
  cs=np.asarray(project(jnp.asarray(raw)))
  h=rollout(start,jnp.asarray(cs))
  eu=float(jnp.sqrt(jnp.mean((np.asarray(h[0][-1])-np.asarray(target[0]))**2)))
  eq=float(jnp.sqrt(jnp.mean((np.asarray(h[1][-1])-np.asarray(target[1]))**2)))
  spatial=np.asarray(cs)@basis.T
  maxf=float(np.max(np.abs(spatial)))
  maxrate=float(np.max(np.abs(np.diff(spatial,axis=0)))/DT)
  stages.append({"weight":weight,"success":bool(opt.success),"message":str(opt.message),
   "iterations":int(opt.nit),"objective":float(opt.fun),"endpoint_u_rms":eu,
   "endpoint_q_rms":eq,"max_abs_f_space":maxf,"max_abs_dfdt_space":maxrate})
  print(json.dumps(stages[-1]),flush=True)
 payload={"config":{"NX":NX,"NT":NT,"T":T,"dt":DT,"nu":NU,"tau":TAU,
  "Fmax_pointwise":FMAX,"rate_max_pointwise":RATE_MAX,
  "projection":"each time step: scale real-space delta field to rate bound, then scale real-space force field to amplitude bound",
  "target":"sample 1 initial field; source is zero-control terminal field from sample 0",
  "objective":"continuation of soft normalized endpoint mismatch plus small actuator effort/slew terms; not exact endpoint constrained and not a certified global minimum"},
  "stages":stages,"elapsed_seconds":time.time()-t0}
 path="/home/iknowyou/TDT_Experimentos/phase6_spatial_homotopy.json"
 with open(path,"w") as f:json.dump(payload,f,indent=2)
 print("SAVED",path)
if __name__=="__main__":main()
