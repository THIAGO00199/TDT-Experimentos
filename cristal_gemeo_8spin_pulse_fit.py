#!/usr/bin/env python3
import json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import least_squares

N=8; NS=1<<N; J=1.0; KT=1.0; beta=1.0; gamma=1.0; hf=1.0
ids=np.arange(NS)
S=(2*((ids[:,None]>>np.arange(N))&1)-1).astype(float)
M=S.sum(axis=1)
bonds=np.stack([S[:,i]*S[:,(i+1)%N] for i in range(N)],axis=1)
Eint=-J*bonds.sum(axis=1)
def gibbs(fields):
    H=Eint-S@np.asarray(fields)
    z=-beta*H/KT; z-=z.max()
    p=np.exp(z); return p/p.sum()
pA=gibbs(np.full(N,hf))
Hf=Eint-hf*M
targetE=float(pA@Hf); targetM=float(pA@M)
# Mathematical B target, alpha=1.5, same E and M.
def normlw(x):
    x=x-x.max(); y=np.exp(x); return y/y.sum()
def pBmath(alpha=1.5):
    def pfun(th):
        b,l=th; return normlw(-b*Eint/KT+l*M+alpha*bonds[:,0])
    def f(th):
        p=pfun(th); return np.array([(p@Hf-targetE)/8,(p@M-targetM)/8])
    from scipy.optimize import root
    so=root(f,[1.0,1.0],tol=1e-11)
    return pfun(so.x),so
pBm,sol=root if False else (None,None)
pBm,sol=pBmath(1.5)
target_corr=pBm@bonds
# Generator for Glauber spin flips; column convention dp/dt=Q p.
def generator(fields):
    rows=[]; cols=[]; vals=[]
    for x in range(NS):
        sx=S[x]
        for i in range(N):
            nb=sx[(i-1)%N]+sx[(i+1)%N]
            dE=2*sx[i]*(J*nb+fields[i])
            rate=gamma/(1.0+np.exp(np.clip(beta*dE/KT,-700,700)))
            y=x^(1<<i)
            rows.append(y); cols.append(x); vals.append(rate)
            rows.append(x); cols.append(x); vals.append(-rate)
    return coo_matrix((vals,(rows,cols)),shape=(NS,NS)).tocsr()
Qbase=generator(np.full(N,hf))
def evolve(params,return_p=False):
    g0,g1,tp,tw=params
    fields=np.full(N,hf); fields[0]+=g0; fields[1]+=g1
    Qp=generator(fields)
    pp=expm_multiply(Qp*tp,pA)
    pf=expm_multiply(Qbase*tw,pp)
    if return_p: return pf,pp
    return pf
def obs(p):
    c=p@bonds
    return np.array([p@Hf,p@M,c[0],np.mean(c[1:])],float)
target=np.array([targetE,targetM,target_corr[0],np.mean(target_corr[1:])])
scales=np.array([8.0,8.0,0.01,0.01])
def residual(x):
    try:
        p=evolve(x)
        return (obs(p)-target)/scales
    except Exception:
        return np.ones(4)*1e3
starts=[
 [1.0,1.0,0.5,1.0], [2.0,2.0,1.0,2.0], [-1.0,-1.0,1.0,2.0],
 [2.0,-2.0,0.5,1.0], [-2.0,2.0,0.5,1.0], [3.0,1.0,2.0,4.0],
 [-1.0,2.0,2.0,5.0], [1.5,-1.5,1.5,3.0], [4.0,4.0,0.25,0.5],
]
runs=[]
t0=time.time()
for k,x0 in enumerate(starts):
    fit=least_squares(residual,x0,bounds=([-6,-6,0.02,0.02],[6,6,8,40]),
       max_nfev=180,xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=2e-4)
    p,pp=evolve(fit.x,True); ob=obs(p)
    runs.append({"start":x0,"x":fit.x.tolist(),"cost":float(fit.cost),
      "optimality":float(fit.optimality),"success":bool(fit.success),
      "nfev":int(fit.nfev),"residual_scaled":residual(fit.x).tolist(),
      "delta_E":float(ob[0]-targetE),"delta_M":float(ob[1]-targetM),
      "corr_selected":float(ob[2]),"corr_other_mean":float(ob[3]),
      "corr_contrast":float(ob[2]-ob[3]),
      "target_corr_selected":float(target[2]),"target_corr_other_mean":float(target[3]),
      "L1_to_math_B":float(np.abs(p-pBm).sum()),"L1_to_A":float(np.abs(p-pA).sum()),
      "Fneq":float(p@Hf+np.sum(p[p>0]*np.log(p[p>0]))),
      "distribution":p.tolist()})
    print("FIT",k,json.dumps({k:v for k,v in runs[-1].items() if k!="distribution"}),flush=True)
best=min(runs,key=lambda x:x["cost"])
out={"model":{"N":N,"states":NS,"J":J,"kBT":KT,"hf":hf,"dynamics":"Glauber single-spin-flip CTMC",
 "pulse":"constant extra local fields on adjacent sites 0 and 1, then removed; baseline hf restored during wait",
 "state_convention":"p column; dp/dt=Qp"},
 "history_A":{"mean_E":targetE,"mean_M":targetM,"corr":(pA@bonds).tolist(),"Fneq":float(pA@Hf+np.sum(pA*np.log(pA)))},
 "math_B":{"alpha":1.5,"mean_E":float(pBm@Hf),"mean_M":float(pBm@M),"corr":target_corr.tolist(),"Fneq":float(pBm@Hf+np.sum(pBm*np.log(pBm)))},
 "fit_runs":runs,"best":{k:v for k,v in best.items() if k!="distribution"},
 "elapsed_seconds":time.time()-t0,
 "note":"This tests physical reachability using a two-site local-field pulse followed by relaxation under hf=1. A good four-observable fit would match E,M and two bond-correlation summary statistics; it does not guarantee the full B distribution is matched."}
path="/home/iknowyou/TDT_Experimentos/cristal_gemeo_8spin_pulse_fit.json"
with open(path,"w") as f: json.dump(out,f,indent=2)
print("SUMMARY",json.dumps({k:v for k,v in out.items() if k not in ("fit_runs","math_B","history_A")},indent=2),flush=True)
