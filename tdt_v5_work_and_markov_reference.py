#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,kron,identity
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
OUT=str(Path(__file__).resolve().parent)
N=3; D=36
wq1,wq2=1.,1.08; wa1,wa2=.82,1.18; g1=g2=.32; J=.025; T=18.
def op(a): return csr_matrix(a,dtype=complex)
def emb(a,site):
 dims=[2,2,N,N]; z=None
 for k,dd in enumerate(dims):
  v=a if k==site else identity(dd,format='csr',dtype=complex)
  z=v if z is None else kron(z,v,format='csr')
 return z
sp=op([[0,1],[0,0]]); sm=sp.getH(); num=sm@sp
a=csr_matrix(np.diag(np.sqrt(np.arange(1,N)),1).astype(complex)); ad=a.getH(); n=ad@a
q1,q2=emb(num,0),emb(num,1); q1p,q2p=emb(sp,0),emb(sp,1)
m1,m2=emb(n,2),emb(n,3); a1,a2=emb(a,2),emb(a,3)
I=identity(D,format='csr',dtype=complex)
H=(wq1*q1+wq2*q2+wa1*m1+wa2*m2+J*(2*q1-I)@(2*q2-I)+g1*(q1p@a1+q1p.getH()@a1.getH())+g2*(q2p@a2+q2p.getH()@a2.getH())).tocsr()
X=op([[0,1],[1,0]]); Y=op([[0,-1j],[1j,0]])
DR=[emb(X,0),emb(Y,0),emb(X,1),emb(Y,1)]
bell=np.zeros(4,complex); bell[0]=bell[3]=2**-.5
vac=np.zeros(9,complex); vac[0]=1
psi0=np.kron(bell,vac)
def rq(x):
 z=x.reshape(4,9); return z@z.conj().T
def ent(r):
 e=np.clip(np.linalg.eigvalsh((r+r.conj().T)/2),0,1); return float(-np.sum(e*np.log(np.maximum(e,1e-15))))
def pur(r): return float(np.real(np.trace(r@r)))
def run_closed(nt,us):
 dt=T/nt; x=psi0.copy(); W=0.
 uold=np.zeros(4)
 for j in range(nt):
  un=us[j]
  # instantaneous quench at boundary: work evaluated in state immediately BEFORE changing field
  for k in range(4):
   du=un[k]-uold[k]
   if du: W += float(np.real(np.vdot(x,DR[k]@x)))*du
  h=H
  for k in range(4):
   if un[k]: h=h+un[k]*DR[k]
  x=expm_multiply((-1j*dt)*h,x); uold=un
 # return controls to zero at final boundary
 for k in range(4):
  du=-uold[k]
  if du: W += float(np.real(np.vdot(x,DR[k]@x)))*du
 Ef=float(np.real(np.vdot(x,H@x))); E0=float(np.real(np.vdot(psi0,H@psi0)))
 return x,W,Ef-E0,W-(Ef-E0)
# Original optimization grid dt=.1 and identical RNG stream/algorithm
nt=180; dt=T/nt
# BLP witness
pplus=np.array([1,0,1,0],complex)/np.sqrt(2); pminus=np.array([1,0,-1,0],complex)/np.sqrt(2)
pa=np.kron(pplus,vac); pb=np.kron(pminus,vac); ds=[1.]
for j in range(nt):
 pa=expm_multiply(-1j*dt*H,pa); pb=expm_multiply(-1j*dt*H,pb)
 za,zb=pa.reshape(4,9),pb.reshape(4,9)
 ds.append(float(np.sum(np.abs(np.linalg.eigvalsh(za@za.conj().T-zb@zb.conj().T)))/2))
dd=np.diff(ds); pos=np.where(dd>1e-10)[0]; s,e=int(pos[0]),int(pos[-1]+1)
K=5; ncoef=20; bound=2.
kn=np.linspace(s,e,K)
def controls(c,nt):
 grid=np.arange(nt); ss=int(round(s*nt/180)); ee=int(round(e*nt/180))
 us=np.zeros((nt,4)); cc=c.reshape(4,K)
 knots=np.linspace(ss,ee,K)
 for k in range(4): us[:,k]=np.interp(grid,knots,cc[k],left=0,right=0)
 us[:ss]=0; us[ee:]=0
 return us
def evolve_entropy(c):
 us=controls(c,180); x=psi0.copy()
 for j in range(180):
  h=H
  for k in range(4):
   if us[j,k]: h=h+us[j,k]*DR[k]
  x=expm_multiply(-1j*dt*h,x)
 r=rq(x); effort=dt*np.sum(us*us)
 return ent(r)+1e-4*effort,ent(r),pur(r)
rng=np.random.default_rng(20261114); c=np.zeros(20); best=c.copy(); bv=evolve_entropy(c)[0]
for _ in range(20):
 cand=np.clip(rng.normal(0,.45,20),-bound,bound); val=evolve_entropy(cand)[0]
 if val<bv: bv=val; best=cand.copy()
c=best.copy(); aS=.16; cS=.20; A=12
for it in range(1,121):
 ak=aS/(it+A)**.602; ck=cS/(it**.101); delta=rng.choice([-1.,1.],20)
 yp=evolve_entropy(np.clip(c+ck*delta,-bound,bound))[0]
 ym=evolve_entropy(np.clip(c-ck*delta,-bound,bound))[0]
 grad=(yp-ym)/(2*ck*delta); c=np.clip(c-ak*grad,-bound,bound)
 val=evolve_entropy(c)[0]
 if val<bv: bv=val; best=c.copy()
us180=controls(best,180)
x,W,dE,res=run_closed(180,us180)
refinements=[]
for nt2 in (360,720):
 us2=controls(best,nt2)
 xx,ww,de,rr=run_closed(nt2,us2)
 refinements.append({'NT':nt2,'dt':T/nt2,'W':ww,'delta_E':de,'residual':rr,'purity':pur(rq(xx)),'entropy':ent(rq(xx))})
r=rq(x)
# Markovian baseline: calibrate zero-temperature amplitude damping rate so no-control purity matches non-Markovian free final purity.
# Qubit-only Hamiltonian and same local X/Y drives. Lindblad jump sigma_- on each qubit.
Dq=4
def qemb(a,site):
 z=None
 for k in range(2):
  v=a if k==site else identity(2,format='csr',dtype=complex)
  z=v if z is None else kron(z,v,format='csr')
 return z
nq=qemb(sm@sp,0); nq2=qemb(sm@sp,1); qp=qemb(sp,0); qp2=qemb(sp,1)
Hq=(wq1*nq+wq2*nq2+J*(2*nq-identity(4,format='csr'))@(2*nq2-identity(4,format='csr'))).toarray()
Dqops=[qemb(X,0).toarray(),qemb(Y,0).toarray(),qemb(X,1).toarray(),qemb(Y,1).toarray()]
Lops=[qemb(sm,0).toarray(),qemb(sm,1).toarray()]
rho0=np.outer(bell,bell.conj())
def liouvillian(gamma, h):
 I4=np.eye(4); L=-1j*(np.kron(I4,h)-np.kron(h.T,I4))
 for c0 in Lops:
  c=np.sqrt(gamma)*c0; cd=c.conj().T; cdc=cd@c
  L+=np.kron(c.conj(),c)-.5*np.kron(I4,cdc)-.5*np.kron(cdc.T,I4)
 return L
# Calibrate gamma by bisection to reproduce free Markovian purity at T=18.
target_free=pur(rq(__import__('functools').reduce(lambda xx,j: expm_multiply(-1j*dt*H,xx),range(180),psi0)))
def markov_final(gamma):
 L=liouvillian(gamma,Hq); v=expm_multiply(L*T,rho0.reshape(-1,order='F'))
 rr=v.reshape((4,4),order='F'); return float(np.real(np.trace(rr@rr)))
lo,hi=0.,1.
for _ in range(50):
 mid=(lo+hi)/2
 if markov_final(mid)>target_free: lo=mid
 else: hi=mid
gamma=(lo+hi)/2
# baseline control with piecewise-constant same 20-knot schedule; minimize positive control work + penalty for missing target purity.
# Work is computed at every control quench; target is the achieved memory purity.
ntm=180; dtm=T/ntm; knots=np.linspace(0,ntm-1,5)
def mcontrols(c):
 cc=c.reshape(4,5); grid=np.arange(ntm); u=np.zeros((ntm,4))
 for k in range(4): u[:,k]=np.interp(grid,knots,cc[k],left=0,right=0)
 u[0]=0; u[-1]=0
 return u
def markov_eval(c):
 u=mcontrols(c); rho=rho0.copy(); Wm=0.; old=np.zeros(4)
 for j in range(ntm):
  for k in range(4):
   du=u[j,k]-old[k]
   if du: Wm+=float(np.real(np.trace(rho@Dqops[k])))*du
  h=Hq.copy()
  for k in range(4): h+=u[j,k]*Dqops[k]
  L=liouvillian(gamma,h)
  rho=(expm(L*dtm)@rho.reshape(-1,order='F')).reshape((4,4),order='F')
  old=u[j].copy()
 for k in range(4):
  du=-old[k]
  if du: Wm+=float(np.real(np.trace(rho@Dqops[k])))*du
 pp=float(np.real(np.trace(rho@rho)))
 # Minimize signed work plus penalty if target purity not reached; small quadratic regularizer.
 cost=Wm+200.*max(0.,opt_target-pp)**2 + 1e-5*dtm*np.sum(u*u)
 return cost,Wm,pp
opt_target=pur(r)
rngm=np.random.default_rng(314159); cm=np.zeros(20); bm=cm.copy(); bval=markov_eval(cm)[0]
for _ in range(12):
 cand=np.clip(rngm.normal(0,.25,20),-2,2); val=markov_eval(cand)[0]
 if val<bval: bval=val; bm=cand.copy()
cm=bm.copy()
for it in range(1,61):
 ak=.05/(it+10)**.602; ck=.10/(it**.101); de=rngm.choice([-1.,1.],20)
 yp=markov_eval(np.clip(cm+ck*de,-2,2))[0]; ym=markov_eval(np.clip(cm-ck*de,-2,2))[0]
 grad=(yp-ym)/(2*ck*de); cm=np.clip(cm-ak*grad,-2,2)
 if markov_eval(cm)[0]<bval: bval=markov_eval(cm)[0]; bm=cm.copy()
mcost,mwork,mpurity=markov_eval(bm)
result={'nonmarkovian':{'NT':180,'dt':dt,'target_purity':pur(r),'entropy_nats':ent(r),'concurrence_not_recomputed':None,'W_pre_quench':W,'delta_E_total':dE,'residual':res,'refinements_fixed_protocol':refinements,'backflow_Pi_BLP_pair':float(np.maximum(dd,0).sum()),'SPSA_iterations':120,'control_coefficients':best.tolist()},'markovian_reference':{'bath':'zero-temperature amplitude damping, two local sigma_- jumps','gamma_calibrated_to_match_uncontrolled_nonmarkovian_final_purity':gamma,'uncontrolled_purity_at_T':target_free,'target_purity':opt_target,'method':'SPSA 60 iterations; same 4 local X/Y channels and 5 temporal knots; objective signed quench work + penalty for missing target','work_quench_estimate':mwork,'achieved_purity':mpurity,'objective':mcost},'comparison':{'delta_W_TDT_candidate':W-mwork,'warning':'This is not yet a valid TDT signature: Lindblad and explicit-mode models differ physically; gamma calibration matches only one endpoint purity; SPSA is not a certified global minimum; the signed drive work can be negative and excludes actuator overhead.'}}
with open(OUT+'/tdt_v5_work_and_markov_reference_result.json','w') as f: json.dump(result,f,indent=2)
with open(OUT+'/tdt_v5_work_and_markov_reference_result.txt','w') as f: f.write(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
