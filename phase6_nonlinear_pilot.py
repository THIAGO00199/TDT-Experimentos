#!/usr/bin/env python3
import os, json, argparse, time
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import pearsonr

PARSER=argparse.ArgumentParser()
PARSER.add_argument("--n-samples",type=int,default=3)
PARSER.add_argument("--maxiter",type=int,default=80)
PARSER.add_argument("--nx",type=int,default=32)
PARSER.add_argument("--nt",type=int,default=40)
ARGS=PARSER.parse_args()
OUT=os.path.dirname(os.path.abspath(__file__))
SEED=20261112
NX=ARGS.nx; NT=ARGS.nt; T=.8; DT=T/NT
NU=.05; TAU=.25
SIG_U=.0015; SIG_Q=.0008
FMAX=.8; RATE_MAX=8.0
MODES=np.arange(1,10)
K=len(MODES); NC=2*K
x=np.arange(NX)*2*np.pi/NX
dx=2*np.pi/NX
kk=np.fft.fftfreq(NX,d=1/NX)
mask=(np.abs(kk)<=NX/3).astype(float)
ik=1j*kk
cosbasis=np.array([np.cos(m*x) for m in MODES])
sinbasis=np.array([np.sin(m*x) for m in MODES])
basis=np.concatenate([cosbasis,sinbasis],axis=0).T
basis_j=jnp.asarray(basis)
mask_j=jnp.asarray(mask)
ik_j=jnp.asarray(ik)
dx_j=dx

def deriv(v):
    return jnp.fft.ifft(ik_j*jnp.fft.fft(v)).real

def rhs(state, coeff):
    u,q=state
    ux=deriv(u); qx=deriv(q)
    f=basis_j@coeff
    adv=u*ux
    adv=jnp.fft.ifft(jnp.fft.fft(adv)*mask_j).real
    return (-adv+qx+f, (NU*ux-q)/TAU)

def rk4(state, coeff):
    a=rhs(state,coeff)
    b=rhs((state[0]+.5*DT*a[0],state[1]+.5*DT*a[1]),coeff)
    c=rhs((state[0]+.5*DT*b[0],state[1]+.5*DT*b[1]),coeff)
    d=rhs((state[0]+DT*c[0],state[1]+DT*c[1]),coeff)
    return (state[0]+DT*(a[0]+2*b[0]+2*c[0]+d[0])/6,
            state[1]+DT*(a[1]+2*b[1]+2*c[1]+d[1])/6)

def rollout(z0, controls):
    def step(state, coeff):
        new=rk4(state,coeff)
        return new, new
    _,hist=jax.lax.scan(step,z0,controls)
    return hist

def make_objective(target_u,target_q,start_u,start_q):
    target=(jnp.asarray(target_u),jnp.asarray(target_q))
    start=(jnp.asarray(start_u),jnp.asarray(start_q))
    # Scaled endpoint norm, with actuator effort and slew regularization.
    scale_u=max(float(np.sqrt(np.mean(target_u**2))),.1)
    scale_q=max(float(np.sqrt(np.mean(target_q**2))),.02)
    def objective(flat):
        c=flat.reshape((NT,NC))
        hist=rollout(start,c)
        uf,qf=hist[0][-1],hist[1][-1]
        endpoint=jnp.mean((uf-target[0])**2)/(scale_u**2)+jnp.mean((qf-target[1])**2)/(scale_q**2)
        effort=jnp.mean(c*c)
        slew=jnp.mean(jnp.diff(c,axis=0)**2)/(DT**2)
        violation=jnp.mean(jax.nn.relu(jnp.abs(jnp.diff(c,axis=0))-RATE_MAX*DT)**2)
        return 2e3*endpoint+2e-4*effort+2e-3*slew+1e3*violation
    return jax.jit(objective),jax.jit(jax.grad(objective))

def physical_metrics(start_u,start_q,controls):
    state=(jnp.asarray(start_u),jnp.asarray(start_q))
    W=0.; D=0.
    for c in controls:
        u=np.asarray(state[0]); q=np.asarray(state[1])
        f=basis@c
        W+=DT*dx*np.sum(u*f)
        D+=DT*dx*np.sum(q*q)/NU
        state=rk4(state,jnp.asarray(c))
    uf=np.asarray(state[0]); qf=np.asarray(state[1])
    def H(u,q):
        return .5*dx*np.sum(u*u)+TAU/(2*NU)*dx*np.sum(q*q)
    H0=H(start_u,start_q); HT=H(uf,qf)
    residual=W-(HT-H0)-D
    return dict(W_net=float(W),Dq=float(D),H_start=float(H0),H_end=float(HT),
                balance_residual=float(residual),u_end=uf,q_end=qf)

def forward(u,q):
    z=(jnp.asarray(u),jnp.asarray(q))
    for _ in range(NT): z=rk4(z,jnp.zeros(NC))
    return np.asarray(z[0]),np.asarray(z[1])

def main():
    rng=np.random.default_rng(SEED)
    Z=rng.standard_normal((100,24))
    samples=[]
    for i in range(100):
        u=np.sin(x).copy(); q=np.zeros(NX)
        for j,m in enumerate(range(4,10)):
            u += SIG_U*(Z[i,2*j]*np.cos(m*x)+Z[i,2*j+1]*np.sin(m*x))
            q += SIG_Q*(Z[i,12+2*j]*np.cos(m*x)+Z[i,13+2*j]*np.sin(m*x))
        samples.append((u,q))
    finals=[forward(u,q) for u,q in samples]
    # Reference full terminal field: trajectory 0, used as the common reconstruction start.
    start_u,start_q=finals[0]
    surprise=.5*np.sum(Z[:ARGS.n_samples]**2,axis=1)+12*np.log(2*np.pi)
    results=[]
    t0=time.time()
    for i in range(ARGS.n_samples):
        target_u,target_q=samples[i]
        fun,grad=make_objective(target_u,target_q,start_u,start_q)
        x0=np.zeros((NT,NC))
        bounds=[(-FMAX,FMAX)]*(NT*NC)
        def f(v): return float(fun(jnp.asarray(v)))
        def g(v): return np.asarray(grad(jnp.asarray(v)),dtype=float)
        opt=minimize(f,x0.ravel(),jac=g,method="L-BFGS-B",bounds=bounds,
                     options={"maxiter":ARGS.maxiter,"ftol":1e-10,"gtol":1e-7,"maxls":20})
        c=opt.x.reshape(NT,NC)
        metrics=physical_metrics(start_u,start_q,c)
        # endpoint error
        end_u,end_q=metrics.pop("u_end"),metrics.pop("q_end")
        err_u=float(np.sqrt(np.mean((end_u-target_u)**2)))
        err_q=float(np.sqrt(np.mean((end_q-target_q)**2)))
        max_slew=float(np.max(np.abs(np.diff(c,axis=0)))/DT)
        metrics.update(sample=i,success=bool(opt.success),message=str(opt.message),
                       iterations=int(opt.nit),objective=float(opt.fun),
                       endpoint_u_rms=err_u,endpoint_q_rms=err_q,
                       max_abs_control=float(np.max(np.abs(c))),
                       max_slew_rate=max_slew,
                       surprisal=float(surprise[i]),
                       initial_energy=float(.5*dx*np.sum(target_u**2)+TAU/(2*NU)*dx*np.sum(target_q**2)))
        results.append(metrics)
        print(f"sample {i+1}/{ARGS.n_samples}: success={opt.success} J={opt.fun:.4g} err_u={err_u:.3g} err_q={err_q:.3g} D={metrics['Dq']:.4g} W={metrics['W_net']:.4g}",flush=True)
    payload=dict(config=dict(seed=SEED,n_samples=ARGS.n_samples,NX=NX,NT=NT,T=T,dt=DT,nu=NU,tau=TAU,
                 sigma_u=SIG_U,sigma_q=SIG_Q,control_modes=MODES.tolist(),fmax=FMAX,rate_max=RATE_MAX,
                 objective="terminal mismatch + actuator effort + slew penalty; bounded coefficient amplitudes; slew enforced by penalty, not hard constraint",
                 common_start="full terminal field of forward trajectory 0"),
                 elapsed_seconds=time.time()-t0,results=results)
    path=os.path.join(OUT,f"phase6_nonlinear_pilot_{ARGS.n_samples}.json")
    with open(path,"w") as f: json.dump(payload,f,indent=2)
    print("SAVED",path,flush=True)
if __name__=="__main__": main()
