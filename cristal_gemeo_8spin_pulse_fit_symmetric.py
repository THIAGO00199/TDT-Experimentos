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
def froot(th,a=1.5):
 p=pb(th,a); return [(p@H-targetE)/8,(p@M-targetM)/8]
s=root(froot,[1,1],tol=1e-12); pB=pb(s.x)
targetC=float(pB@bonds[:,0]); targetCorr=pB@bonds
def gen(g):
 fields=np.full(N,hf); fields[0]+=g; fields[1]+=g
 r=[];c=[];v=[]
 for x in range(NS):
  sx=S[x]
  for i in range(N):
   dE=2*sx[i]*(J*(sx[(i-1)%N]+sx[(i+1)%N])+fields[i])
   rate=1/(1+np.exp(np.clip(dE,-700,700)))
   y=x^(1<<i); r.extend([y,x]);c.extend([x,x]);v.extend([rate,-rate])
 return coo_matrix((v,(r,c)),shape=(NS,NS)).tocsr()
Q0=gen(0.)
def evolve(x):
 g,tp,tw=x
 p=expm_multiply(gen(g)*tp,pA)
 return expm_multiply(Q0*tw,p)
def obs(p):
 c=p@bonds
 return np.array([p@H,p@M,c[0]])
scale=np.array([8.,8.,.01]); targ=np.array([targetE,targetM,targetC])
def res(x): return (obs(evolve(x))-targ)/scale
starts=[]
for g in [-8,-5,-3,-1,1,3,5,8]:
 for tp in [.05,.5,2.]:
  starts.append([g,tp,1.])
runs=[];t0=time.time()
for j,x0 in enumerate(starts):
 z=least_squares(res,x0,bounds=([-10,.005,.005],[10,12,50]),max_nfev=250,
                  xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=1e-4)
 p=evolve(z.x); o=obs(p); cc=p@bonds
 row={"start":x0,"x":z.x.tolist(),"cost":float(z.cost),"nfev":int(z.nfev),
 "res":res(z.x).tolist(),"dE":float(o[0]-targetE),"dM":float(o[1]-targetM),
 "dC0":float(o[2]-targetC),"corr_contrast":float(cc[0]-np.mean(cc[1:])),
 "target_contrast":float(targetCorr[0]-np.mean(targetCorr[1:])),
 "L1B":float(np.abs(p-pB).sum()),"L1A":float(np.abs(p-pA).sum()),
 "Fneq":float(p@H+np.sum(p[p>0]*np.log(p[p>0])))}
 runs.append(row)
 if j%4==0: print("FIT",j,json.dumps(row),flush=True)
best=min(runs,key=lambda q:q["cost"])
out={"target":{"alpha":1.5,"E":targetE,"M":targetM,"C0":targetC,"corr":targetCorr.tolist(),
 "Fneq_A":float(pA@H+np.sum(pA*np.log(pA))),"Fneq_B":float(pB@H+np.sum(pB*np.log(pB)))},
 "runs":runs,"best":best,"elapsed_s":time.time()-t0}
with open("/home/iknowyou/TDT_Experimentos/cristal_gemeo_8spin_pulse_fit_symmetric.json","w") as f: json.dump(out,f,indent=2)
print("BEST",json.dumps(best),flush=True)
