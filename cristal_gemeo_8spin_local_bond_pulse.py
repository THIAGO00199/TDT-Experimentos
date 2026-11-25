#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import least_squares,root
N=8; NS=256; J=1.; KT=1.; beta=1.; gamma=1.; hf=1.
ids=np.arange(NS); S=(2*((ids[:,None]>>np.arange(N))&1)-1).astype(float)
M=S.sum(1); bonds=np.stack([S[:,i]*S[:,(i+1)%N] for i in range(N)],1)
Eint=-J*bonds.sum(1); H=Eint-hf*M
def soft(x):
 x=x-x.max(); p=np.exp(x); return p/p.sum()
pA=soft(-H)
targetE=float(pA@H); targetM=float(pA@M)
def pb(th,a=1.5):
 b,l=th; return soft(-b*Eint+l*M+a*bonds[:,0])
def ff(th,a=1.5):
 p=pb(th,a); return [(p@H-targetE)/8,(p@M-targetM)/8]
s=root(ff,[1,1],tol=1e-12); pB=pb(s.x)
targetC=float(pB@bonds[:,0]); targetCorr=pB@bonds
def gen(deltaJ=0.):
 jb=np.ones(N); jb[0]+=deltaJ
 rows=[];cols=[];vals=[]
 for x in range(NS):
  sx=S[x]
  for i in range(N):
   loc=jb[(i-1)%N]*sx[(i-1)%N]+jb[i]*sx[(i+1)%N]+hf
   dE=2*sx[i]*loc
   rate=gamma/(1+np.exp(np.clip(beta*dE/KT,-700,700)))
   y=x^(1<<i); rows.extend([y,x]);cols.extend([x,x]);vals.extend([rate,-rate])
 return coo_matrix((vals,(rows,cols)),shape=(NS,NS)).tocsr()
Q0=gen(0.)
def ev(x):
 dj,tp,tw=x
 pp=expm_multiply(gen(dj)*tp,pA)
 return expm_multiply(Q0*tw,pp)
def obs(p): return np.array([p@H,p@M,(p@bonds)[0]])
targ=np.array([targetE,targetM,targetC]); scale=np.array([8.,8.,.01])
def res(x): return (obs(ev(x))-targ)/scale
starts=[]
for dj in [0.5,1,2,4,8]:
 for tp in [.05,.25,1.]:
  starts.append([dj,tp,1.])
for dj in [-1,-3,-6]:
 starts.append([dj,.5,.5])
runs=[];t0=time.time()
for j,x0 in enumerate(starts):
 z=least_squares(res,x0,bounds=([-10,.005,.005],[20,12,40]),max_nfev=180,
                  xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=1e-4)
 p=ev(z.x); o=obs(p); cc=p@bonds
 row={"start":x0,"x":z.x.tolist(),"cost":float(z.cost),"nfev":int(z.nfev),
 "res":res(z.x).tolist(),"dE":float(o[0]-targetE),"dM":float(o[1]-targetM),
 "dC0":float(o[2]-targetC),"corr_contrast":float(cc[0]-np.mean(cc[1:])),
 "target_contrast":float(targetCorr[0]-np.mean(targetCorr[1:])),
 "L1B":float(np.abs(p-pB).sum()),"L1A":float(np.abs(p-pA).sum()),
 "Fneq":float(p@H+np.sum(p[p>0]*np.log(p[p>0])))}
 runs.append(row)
 print("FIT",j,json.dumps(row),flush=True)
best=min(runs,key=lambda q:q["cost"])
out={"protocol":"local interaction pulse on bond (0,1): J01=J+deltaJ for tp, then baseline J and Glauber relaxation tw",
 "target":{"alpha":1.5,"E":targetE,"M":targetM,"C0":targetC,"corr":targetCorr.tolist(),
 "Fneq_A":float(pA@H+np.sum(pA*np.log(pA))),"Fneq_B":float(pB@H+np.sum(pB*np.log(pB)))},
 "runs":runs,"best":best,"elapsed_s":time.time()-t0}
with open("/home/iknowyou/TDT_Experimentos/cristal_gemeo_8spin_local_bond_pulse.json","w") as f: json.dump(out,f,indent=2)
print("BEST",json.dumps(best),flush=True)
