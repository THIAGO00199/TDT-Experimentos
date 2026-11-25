#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply

OUT='/home/iknowyou/TDT_Experimentos'
T=18.0; NT=180; dt=T/NT; K=9; BOUND=1.5
TARGET=.85; GAMMA=.005
I2=np.eye(2,dtype=complex); I4=np.eye(4,dtype=complex)
sp=np.array([[0,1],[0,0]],complex); sm=sp.conj().T
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex)
def emb2(a,s): return np.kron(a,I2) if s==0 else np.kron(I2,a)
n1=emb2(sm@sp,0); n2=emb2(sm@sp,1)
Hq=1.0*n1+1.08*n2+.025*(2*n1-I4)@(2*n2-I4)
Oq=[emb2(X,0),emb2(Y,0),emb2(X,1),emb2(Y,1)]
Cq=[emb2(sm,0),emb2(sm,1)]
bell=np.array([1,0,0,1],complex)/np.sqrt(2); rho0=np.outer(bell,bell.conj())
def D(r):
 out=np.zeros((4,4),complex)
 for c0 in Cq:
  c=np.sqrt(GAMMA)*c0; cd=c.conj().T; cdc=cd@c
  out+=c@r@cd-.5*(cdc@r+r@cdc)
 return out
def rhs(r,H): return -1j*(H@r-r@H)+D(r)
def rk4(r,H,h):
 k1=rhs(r,H); k2=rhs(r+.5*h*k1,H); k3=rhs(r+.5*h*k2,H); k4=rhs(r+h*k3,H)
 return r+(h/6)*(k1+2*k2+2*k3+k4)
# Explicit two-mode closed model, same initial Bell state and local controls.
N=3; DIM=36
def embed4(op,site):
 dims=[2,2,N,N]; z=None
 for j,d in enumerate(dims):
  a=op if j==site else identity(d,format='csr',dtype=complex)
  z=a if z is None else kron(z,a,format='csr')
 return z
def op(a): return csr_matrix(a,dtype=complex)
smsp=op([[0,0],[0,1]]); spq=op([[0,1],[0,0]]); smq=spq.getH()
num=smq@spq
ann=csr_matrix(np.diag(np.sqrt(np.arange(1,N)),1).astype(complex)); adag=ann.getH(); nm=adag@ann
Iq=identity(DIM,format='csr',dtype=complex)
q1n,q2n=embed4(num,0),embed4(num,1)
m1a,m2a=embed4(ann,2),embed4(ann,3)
m1n,m2n=embed4(nm,2),embed4(nm,3)
Hmem=(1.0*q1n+1.08*q2n+.82*m1n+1.18*m2n+.025*(2*q1n-Iq)@(2*q2n-Iq)
 +.32*(embed4(spq,0)@m1a+embed4(spq,0).getH()@m1a.getH())
 +.32*(embed4(spq,1)@m2a+embed4(spq,1).getH()@m2a.getH())).tocsr()
Xsp=op([[0,1],[1,0]]); Ysp=op([[0,-1j],[1j,0]])
Om=[embed4(Xsp,0),embed4(Ysp,0),embed4(Xsp,1),embed4(Ysp,1)]
psi_b=np.zeros(4,complex); psi_b[[0,3]]=1/np.sqrt(2)
vac=np.zeros(N*N,complex); vac[0]=1
psi0=np.kron(psi_b,vac)
grid=np.arange(NT); knots=np.linspace(0,NT-1,K)
def controls(c):
 cc=np.asarray(c).reshape(4,K); u=np.zeros((NT,4))
 for k in range(4): u[:,k]=np.interp(grid,knots,cc[k])
 u[0]=0.; u[-1]=0.
 return u
def sim_markov(c):
 u=controls(c); r=rho0.copy(); W=0.; Q=0.; old=np.zeros(4)
 for j in range(NT):
  for k in range(4):
   du=u[j,k]-old[k]
   if du: W+=float(np.real(np.trace(r@Oq[k])))*du
  H=Hq.copy()
  for k in range(4): H+=u[j,k]*Oq[k]
  rm=rk4(r,H,.5*dt)
  Q+=dt*float(np.real(np.trace(H@D(rm))))
  r=rk4(r,H,dt); old=u[j].copy()
 for k in range(4):
  du=-old[k]
  if du: W+=float(np.real(np.trace(r@Oq[k])))*du
 P=float(np.real(np.trace(r@r))); dE=float(np.real(np.trace(r@Hq))-np.trace(rho0@Hq).real)
 return W,Q,dE,P
def rho_qubits(psi):
 z=psi.reshape(4,N*N); return z@z.conj().T
def sim_memory(c):
 u=controls(c); x=psi0.copy(); W=0.; old=np.zeros(4)
 for j in range(NT):
  for k in range(4):
   du=u[j,k]-old[k]
   if du: W+=float(np.real(np.vdot(x,Om[k]@x)))*du
  H=Hmem.copy()
  for k in range(4):
   if u[j,k]!=0: H=H+u[j,k]*Om[k]
  x=expm_multiply((-1j*dt)*H,x)
  old=u[j].copy()
 for k in range(4):
  du=-old[k]
  if du: W+=float(np.real(np.vdot(x,Om[k]@x)))*du
 rq=rho_qubits(x); P=float(np.real(np.trace(rq@rq)))
 E0=float(np.real(np.vdot(psi0,Hmem@psi0))); Ef=float(np.real(np.vdot(x,Hmem@x)))
 return W,0.,Ef-E0,P
def spsa(fun,seed,niter,scale=.12):
 rng=np.random.default_rng(seed); best=None
 starts=[np.zeros(4*K)]+[np.clip(rng.normal(0,.35,4*K),-BOUND,BOUND) for _ in range(5)]
 for c in starts:
  val=fun(c)
  if best is None or val<best[0]: best=(val,c.copy())
 c=best[1].copy()
 for it in range(1,niter+1):
  ak=scale/(it+10)**.602; ck=.12/(it**.101)
  d=rng.choice([-1.,1.],4*K)
  yp=fun(np.clip(c+ck*d,-BOUND,BOUND)); ym=fun(np.clip(c-ck*d,-BOUND,BOUND))
  grad=(yp-ym)/(2*ck*d)
  c=np.clip(c-ak*grad,-BOUND,BOUND)
  val=fun(c)
  if val<best[0]: best=(val,c.copy())
 return best
# Markovian P-max: maximize purity directly; this is a best-found value, not a proven ceiling.
t0=time.time(); pmax_records=[]
for seed in range(20):
 v,c=spsa(lambda z:-sim_markov(z)[3],20261115+seed,300,.18)
 met=sim_markov(c); pmax_records.append({'seed':seed,'purity':met[3],'W_ctrl':met[0],'Q':met[1],'delta_E':met[2]})
 print('PMAX_SEED',seed,'P',met[3],flush=True)
best_pmax=max(pmax_records,key=lambda z:z['purity'])

result={'K':K,'T':T,'NT':NT,'dt':dt,'control_bound':BOUND,
        'method':'SPSA direct purity maximization; 20 seeds x 300 iterations; six initial candidates per seed',
        'seed_base':20261115,'seeds':pmax_records,'best_found':best_pmax,
        'mean_purity':float(np.mean([r['purity'] for r in pmax_records])),
        'std_purity':float(np.std([r['purity'] for r in pmax_records],ddof=1)),
        'certified_upper_bound':False}
path=OUT+'/tdt_v5_pmax_K9_20seeds_300iter.json'
with open(path,'w') as f: json.dump(result,f,indent=2)
with open(OUT+'/tdt_v5_pmax_K9_20seeds_300iter.txt','w') as f: f.write(json.dumps(result,indent=2))
print('CONSOLIDATED_RESULT',json.dumps(result),flush=True)
