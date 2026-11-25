#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import phase6_projected_subspace as m

OUT=Path("/home/iknowyou/TDT_Experimentos/phase6_separated_results.json")
WEIGHTS=[1e2,1e3,1e4,1e5]
MAXITER=500
NSTART=3

def run_one(kind,start,target,seed):
    su=max(float(jnp.sqrt(jnp.mean(target[0]**2))),.1)
    sq=max(float(jnp.sqrt(jnp.mean(target[1]**2))),.02)
    def obj(flat,weight):
        raw=flat.reshape(m.NT,m.NC)
        cs=m.project(raw)
        h=m.rollout(start,cs)
        eu=jnp.mean(((h[0][-1]-target[0])/su)**2)
        eq=jnp.mean(((h[1][-1]-target[1])/sq)**2)
        err=eu if kind=="u_only" else eq
        effort=jnp.mean(cs**2)
        slew=jnp.mean(jnp.diff(cs,axis=0)**2)/(m.DT**2)
        return weight*err+1e-5*effort+1e-7*slew
    stages=[]
    rng=np.random.default_rng(seed)
    raw=np.zeros((m.NT,m.NC)) if seed==0 else rng.normal(0,.15,(m.NT,m.NC))
    for weight in WEIGHTS:
        fun=jax.jit(lambda v:obj(v,weight))
        grad=jax.jit(jax.grad(lambda v:obj(v,weight)))
        opt=minimize(lambda v:float(fun(jnp.asarray(v))),raw.ravel(),
            jac=lambda v:np.asarray(grad(jnp.asarray(v)),float),
            method="L-BFGS-B",bounds=[(-20,20)]*(m.NT*m.NC),
            options={"maxiter":MAXITER,"ftol":1e-10,"gtol":1e-7,"maxls":30,"maxcor":15})
        raw=opt.x.reshape(m.NT,m.NC)
        cs=np.asarray(m.project(jnp.asarray(raw)))
        h=m.rollout(start,jnp.asarray(cs))
        eu=float(jnp.sqrt(jnp.mean((np.asarray(h[0][-1])-np.asarray(target[0]))**2)))
        eq=float(jnp.sqrt(jnp.mean((np.asarray(h[1][-1])-np.asarray(target[1]))**2)))
        spatial=cs@m.basis.T
        rec={"weight":weight,"success":bool(opt.success),"message":str(opt.message),
             "iterations":int(opt.nit),"objective":float(opt.fun),
             "endpoint_u_rms":eu,"endpoint_q_rms":eq,
             "max_abs_f_space":float(np.max(np.abs(spatial))),
             "max_abs_dfdt_space":float(np.max(np.abs(np.diff(spatial,axis=0)))/m.DT)}
        stages.append(rec)
        print(json.dumps({"kind":kind,"seed":seed,"stage":rec}),flush=True)
    return {"seed":seed,"stages":stages}

def grad_check(start,target):
    su=max(float(jnp.sqrt(jnp.mean(target[0]**2))),.1)
    sq=max(float(jnp.sqrt(jnp.mean(target[1]**2))),.02)
    def loss(flat,kind):
        cs=m.project(flat.reshape(m.NT,m.NC))
        h=m.rollout(start,cs)
        if kind=="u_only": return jnp.mean(((h[0][-1]-target[0])/su)**2)
        return jnp.mean(((h[1][-1]-target[1])/sq)**2)
    rng=np.random.default_rng(20261115)
    reports=[]
    for scale,label in [(0.03,"mostly_unsaturated"),(3.0,"saturation_prone")]:
      v=jnp.asarray(rng.normal(0,scale,(m.NT*m.NC,)))
      d=np.asarray(rng.normal(size=v.shape)); d/=np.linalg.norm(d)
      for kind in ["u_only","q_only"]:
        g=np.asarray(jax.grad(lambda z:loss(z,kind))(v))
        rows=[]
        for eps in [1e-4,1e-5,1e-6]:
          fp=float(loss(v+eps*d,kind)); fm=float(loss(v-eps*d,kind))
          fd=(fp-fm)/(2*eps); ad=float(g@d)
          rows.append({"epsilon":eps,"autodiff_directional":ad,
                       "finite_difference_directional":fd,
                       "absolute_error":abs(ad-fd),
                       "relative_error":abs(ad-fd)/max(abs(fd),1e-12)})
        reports.append({"region":label,"kind":kind,"checks":rows})
    return reports

def gramian(start):
    # Orthonormal real Fourier basis: cos/sin k=1..10 for u, then q.
    cols=[]
    for field in range(2):
      for k in range(1,11):
        for fn in (np.cos,np.sin):
          v=np.zeros((2,m.NX)); v[field]=fn(k*m.x)*np.sqrt(2/m.NX)
          cols.append(v.reshape(-1))
    P=np.stack(cols,axis=1) # 64 x 40
    # B maps the 18 actuator coefficients into active coordinates.
    Bfull=np.vstack([m.basis,np.zeros_like(m.basis)])
    B=P.T@Bfull
    def rhs_flat(y):
      u=y[:m.NX]; q=y[m.NX:]
      du,dq=m.rhs((u,q),jnp.zeros(m.NC))
      return jnp.concatenate([du,dq])
    jac=jax.jit(jax.jacfwd(rhs_flat))
    z=start
    G=np.zeros((40,40))
    eigtrack=[]
    for it in range(m.NT):
      y=np.concatenate([np.asarray(z[0]),np.asarray(z[1])])
      A=P.T@np.asarray(jac(jnp.asarray(y)))@P
      def gd(g): return A@g+g@A.T+B@B.T
      k1=gd(G); k2=gd(G+m.DT*k1/2); k3=gd(G+m.DT*k2/2); k4=gd(G+m.DT*k3)
      G=G+m.DT*(k1+2*k2+2*k3+k4)/6
      G=(G+G.T)/2
      z=m.rk4(z,jnp.zeros(m.NC))
      if it in [19,39,59,79]:
        eigtrack.append({"step":it+1,"min_eigenvalue":float(np.linalg.eigvalsh(G)[0]),
                         "max_eigenvalue":float(np.linalg.eigvalsh(G)[-1])})
    vals,vecs=np.linalg.eigh(G)
    order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
    dirs=[]
    for j in range(40):
      v=vecs[:,j]
      contributions=[]
      idx=0
      for field,name in [(0,"u"),(1,"q")]:
        for k in range(1,11):
          a=v[idx]; b=v[idx+1]; idx+=2
          contributions.append((a*a+b*b,name,k))
      contributions.sort(reverse=True)
      dirs.append({"rank":j+1,"eigenvalue":float(vals[j]),
                   "effort_proxy_1_over_sqrt_lambda":float(1/np.sqrt(max(vals[j],1e-300))),
                   "dominant_fourier_components":[{"field":n,"k":k,"energy_fraction":float(e)}
                                                   for e,n,k in contributions[:5]],
                   "eigenvector":[float(x) for x in v]})
    return {"basis":"orthonormal real Fourier modes k=1..10 for u and q",
            "dimension":40,"eigenvalues_descending":[float(v) for v in vals],
            "condition_number":float(vals[0]/vals[-1]) if vals[-1]>0 else None,
            "trajectory_eigenvalue_checkpoints":eigtrack,
            "directions":dirs}

def main():
    t0=time.time()
    s0=m.make_sample(20261112); s1=m.make_sample(20261113)
    start=m.project_state(m.forward(m.project_state((jnp.asarray(s0[0]),jnp.asarray(s0[1])))))
    target=m.project_state((jnp.asarray(s1[0]),jnp.asarray(s1[1])))
    baseline={"u_rms":float(jnp.sqrt(jnp.mean((start[0]-target[0])**2))),
              "q_rms":float(jnp.sqrt(jnp.mean((start[1]-target[1])**2)))}
    result={"config":{"NX":m.NX,"NT":m.NT,"T":m.T,"dt":m.DT,"nu":m.NU,
                      "tau":m.TAU,"weights":WEIGHTS,"maxiter_per_stage":MAXITER,
                      "starts_per_objective":NSTART,
                      "losses":"u_only optimizes normalized u endpoint error; q_only optimizes normalized q endpoint error; other field is diagnostic only",
                      "actuator_bounds":{"spatial_amplitude":m.FMAX,"spatial_rate":m.RATE_MAX}},
            "baseline_no_control":baseline,"optimizations":{}}
    for kind in ["u_only","q_only"]:
      runs=[]
      for seed in range(NSTART):
        runs.append(run_one(kind,start,target,seed))
      result["optimizations"][kind]=runs
    result["gradient_validation"]=grad_check(start,target)
    result["projected_gramian"]=gramian(start)
    result["elapsed_seconds"]=time.time()-t0
    OUT.write_text(json.dumps(result,indent=2))
    print("SAVED",str(OUT),flush=True)
    print("ELAPSED",result["elapsed_seconds"],flush=True)
if __name__=="__main__": main()
