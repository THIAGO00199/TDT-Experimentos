#!/usr/bin/env python3
import os,json,argparse,time
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from scipy.optimize import minimize

ap=argparse.ArgumentParser()
ap.add_argument("--n-samples",type=int,default=3)
ap.add_argument("--maxiter",type=int,default=250)
ap.add_argument("--nx",type=int,default=32)
ap.add_argument("--nt",type=int,default=80)
a=ap.parse_args()
OUT=os.path.dirname(os.path.abspath(__file__))
SEED=20261112; NX=a.nx; NT=a.nt; T=.8; DT=T/NT
NU=.05; TAU=.25; SU=.0015; SQ=.0008
FMAX=2.0; RATE_MAX=10.0
MODES=np.arange(1,10); NC=2*len(MODES)
x=np.arange(NX)*2*np.pi/NX; dx=2*np.pi/NX
kk=np.fft.fftfreq(NX,d=1/NX); ik=1j*kk
mask=(np.abs(kk)<=NX/3).astype(float)
basis=np.stack([f(m*x) for f in (np.cos,np.sin) for m in MODES],axis=0).T
BJ=jnp.asarray(basis); MK=jnp.asarray(mask); IK=jnp.asarray(ik)

def deriv(v): return jnp.fft.ifft(IK*jnp.fft.fft(v)).real
def rhs(z,c):
 u,q=z; ux=deriv(u); qx=deriv(q)
 adv=jnp.fft.ifft(jnp.fft.fft(u*ux)*MK).real
 return (-adv+qx+BJ@c,(NU*ux-q)/TAU)
def rk4(z,c):
 k1=rhs(z,c)
 z2=(z[0]+DT*k1[0]/2,z[1]+DT*k1[1]/2)
 k2=rhs(z2,c)
 z3=(z[0]+DT*k2[0]/2,z[1]+DT*k2[1]/2)
 k3=rhs(z3,c)
 z4=(z[0]+DT*k3[0],z[1]+DT*k3[1])
 k4=rhs(z4,c)
 return (z[0]+DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
         z[1]+DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)

def bounded_controls(raw):
 def step(prev,desired):
  nxt=jnp.clip(prev+jnp.clip(desired-prev,-RATE_MAX*DT,RATE_MAX*DT),-FMAX,FMAX)
  return nxt,nxt
 _,cs=jax.lax.scan(step,jnp.zeros(NC),raw)
 return cs

def rollout(z0,cs):
 def step(z,c):
  zn=rk4(z,c)
  return zn,zn
 _,h=jax.lax.scan(step,z0,cs)
 return h

def objective_factory(tu,tq,su,sq):
 target=(jnp.asarray(tu),jnp.asarray(tq)); start=(jnp.asarray(su),jnp.asarray(sq))
 scaleu=max(float(np.sqrt(np.mean(tu**2))),.1)
 scaleq=max(float(np.sqrt(np.mean(tq**2))),.02)
 def obj(flat):
  raw=flat.reshape(NT,NC); cs=bounded_controls(raw)
  h=rollout(start,cs); ue,qe=h[0][-1],h[1][-1]
  e=jnp.mean((ue-target[0])**2)/scaleu**2+jnp.mean((qe-target[1])**2)/scaleq**2
  effort=jnp.mean(cs**2)
  slew=jnp.mean(jnp.diff(cs,axis=0)**2)/DT**2
  return 2e4*e+1e-4*effort+2e-5*slew
 return jax.jit(obj),jax.jit(jax.grad(obj))

def step_metrics(z,c):
 u,q=z
 k1=rhs(z,c)
 z2=(u+DT*k1[0]/2,q+DT*k1[1]/2); k2=rhs(z2,c)
 z3=(u+DT*k2[0]/2,q+DT*k2[1]/2); k3=rhs(z3,c)
 z4=(u+DT*k3[0],q+DT*k3[1]); k4=rhs(z4,c)
 f=basis@np.asarray(c)
 def rates(zz):
  uu,qq=np.asarray(zz[0]),np.asarray(zz[1])
  return dx*np.sum(uu*f),dx*np.sum(qq*qq)/NU
 w1,d1=rates(z); w2,d2=rates(z2); w3,d3=rates(z3); w4,d4=rates(z4)
 W=DT*(w1+2*w2+2*w3+w4)/6
 D=DT*(d1+2*d2+2*d3+d4)/6
 zn=(u+DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
     q+DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)
 return zn,W,D
def H(u,q): return .5*dx*np.sum(u*u)+TAU/(2*NU)*dx*np.sum(q*q)
def forward(u,q):
 z=(jnp.asarray(u),jnp.asarray(q))
 for _ in range(NT): z=rk4(z,jnp.zeros(NC))
 return np.asarray(z[0]),np.asarray(z[1])
def runmetrics(su,sq,cs):
 z=(jnp.asarray(su),jnp.asarray(sq)); W=D=0.
 for c in cs: z,w,d=step_metrics(z,c); W+=w; D+=d
 ue,qe=np.asarray(z[0]),np.asarray(z[1]); h0=H(su,sq); h1=H(ue,qe)
 return dict(W_net=float(W),Dq=float(D),H_start=float(h0),H_end=float(h1),
             balance_residual=float(W-(h1-h0)-D),
             endpoint_u_rms=float(np.sqrt(np.mean((ue)**2))),
             u_end=ue,q_end=qe)
def main():
 rng=np.random.default_rng(SEED); Z=rng.standard_normal((100,24)); samples=[]
 for i in range(100):
  u=np.sin(x).copy(); q=np.zeros(NX)
  for j,m in enumerate(range(4,10)):
   u+=SU*(Z[i,2*j]*np.cos(m*x)+Z[i,2*j+1]*np.sin(m*x))
   q+=SQ*(Z[i,12+2*j]*np.cos(m*x)+Z[i,13+2*j]*np.sin(m*x))
  samples.append((u,q))
 finals=[forward(u,q) for u,q in samples]
 startu,startq=finals[0]
 surprise=.5*np.sum(Z[:a.n_samples]**2,axis=1)+12*np.log(2*np.pi)
 results=[]; t0=time.time()
 for i in range(a.n_samples):
  tu,tq=samples[i]; fun,grad=objective_factory(tu,tq,startu,startq)
  x0=np.zeros((NT,NC)); bounds=[(-FMAX,FMAX)]*(NT*NC)
  def f(v): return float(fun(jnp.asarray(v)))
  def g(v): return np.asarray(grad(jnp.asarray(v)),float)
  opt=minimize(f,x0.ravel(),jac=g,method="L-BFGS-B",bounds=bounds,
      options={"maxiter":a.maxiter,"ftol":1e-11,"gtol":1e-8,"maxls":30,"maxcor":20})
  raw=opt.x.reshape(NT,NC); cs=np.asarray(bounded_controls(jnp.asarray(raw)))
  met=runmetrics(startu,startq,cs); ue,qe=met.pop("u_end"),met.pop("q_end")
  met.update(sample=i,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),
   objective=float(opt.fun),endpoint_u_error=float(np.sqrt(np.mean((ue-tu)**2))),
   endpoint_q_error=float(np.sqrt(np.mean((qe-tq)**2))),
   max_abs_control=float(np.max(np.abs(cs))),
   max_slew_rate=float(np.max(np.abs(np.diff(cs,axis=0)))/DT),
   surprisal=float(surprise[i]),
   initial_energy=float(H(tu,tq)))
  results.append(met)
  print(f"sample {i+1}/{a.n_samples} success={opt.success} J={opt.fun:.5g} "
        f"erru={met['endpoint_u_error']:.4g} errq={met['endpoint_q_error']:.4g} "
        f"D={met['Dq']:.5g} W={met['W_net']:.5g} residual={met['balance_residual']:.2e} "
        f"slew={met['max_slew_rate']:.3g}",flush=True)
 payload=dict(config=dict(seed=SEED,n_samples=a.n_samples,NX=NX,NT=NT,T=T,dt=DT,nu=NU,tau=TAU,
  sigma_u=SU,sigma_q=SQ,control_modes=MODES.tolist(),fmax=FMAX,rate_max=RATE_MAX,
  common_start="full terminal field of forward trajectory 0",
  optimizer="JAX autodiff + L-BFGS-B; controls are hard amplitude- and slew-limited by sequential projection",
  objective="terminal state mismatch + actuator effort + slew regularization",
  note="pilot only; objective is a soft terminal penalty, not exact endpoint-constrained optimization"),
  elapsed_seconds=time.time()-t0,results=results)
 path=os.path.join(OUT,f"phase6_nonlinear_pilot_v2_{a.n_samples}.json")
 with open(path,"w") as f: json.dump(payload,f,indent=2)
 print("SAVED",path,flush=True)
if __name__=="__main__": main()
