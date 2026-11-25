#!/usr/bin/env python3
import json, time
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, kron, identity

OUT='/home/iknowyou/TDT_Experimentos'
T=18.0; NT=180; dt=T/NT; target=0.935
w1,w2,J=1.0,1.08,0.025
I2=np.eye(2,dtype=complex); I4=np.eye(4,dtype=complex)
sp=np.array([[0,1],[0,0]],complex); sm=sp.conj().T
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex)
def emb(a,s):
 return np.kron(a,I2) if s==0 else np.kron(I2,a)
n1=emb(sm@sp,0); n2=emb(sm@sp,1)
H0=w1*n1+w2*n2+J*(2*n1-I4)@(2*n2-I4)
O=[emb(X,0),emb(Y,0),emb(X,1),emb(Y,1)]
C=[emb(sm,0),emb(sm,1)]
bell=np.array([1,0,0,1],complex)/np.sqrt(2)
rho0=np.outer(bell,bell.conj())
def dissipator(rho,gamma):
 out=np.zeros((4,4),complex)
 for c0 in C:
  c=np.sqrt(gamma)*c0; cd=c.conj().T; cdc=cd@c
  out += c@rho@cd - .5*(cdc@rho+rho@cdc)
 return out
def Lmat(H,gamma):
 # column-major vectorization
 L=-1j*(np.kron(I4,H)-np.kron(H.T,I4))
 for c0 in C:
  c=np.sqrt(gamma)*c0; cd=c.conj().T; cdc=cd@c
  L += np.kron(c.conj(),c)-.5*np.kron(I4,cdc)-.5*np.kron(cdc.T,I4)
 return L
def evolve_free(gamma):
 L=Lmat(H0,gamma)
 v=expm(L*T)@rho0.reshape(-1,order='F')
 r=v.reshape((4,4),order='F')
 return float(np.real(np.trace(r@r)))
# gamma scan, select moderate damping yielding the closest uncontrolled purity below target
scan=[]
for g in [0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.18,0.25]:
 scan.append({'gamma':g,'purity':evolve_free(g)})
eligible=[z for z in scan if z['purity']<target]
gamma=min(eligible,key=lambda z:abs(z['purity']-(target-0.08)))['gamma'] if eligible else 0.03
# controls: four local X/Y channels, five temporal knots; zero at both endpoints
K=5; grid=np.arange(NT); knots=np.linspace(0,NT-1,K)
def make_u(c):
 cc=c.reshape(4,K); u=np.zeros((NT,4))
 for k in range(4): u[:,k]=np.interp(grid,knots,cc[k])
 u[0,:]=0; u[-1,:]=0
 return u
def simulate(c, return_trace=False):
 u=make_u(c); rho=rho0.copy(); W=0.; Q=0.
 old=np.zeros(4)
 for j in range(NT):
  # Quench work at the boundary, using rho immediately before the change
  for k in range(4):
   du=u[j,k]-old[k]
   if du: W += float(np.real(np.trace(rho@O[k])))*du
  H=H0.copy()
  for k in range(4): H += u[j,k]*O[k]
  L=Lmat(H,gamma)
  v=rho.reshape(-1,order='F')
  # midpoint quadrature for heat, with H fixed over this segment
  vm=expm(L*(dt/2))@v
  rm=vm.reshape((4,4),order='F')
  Q += dt*float(np.real(np.trace(H@dissipator(rm,gamma))))
  vf=expm(L*dt)@v
  rho=vf.reshape((4,4),order='F')
  old=u[j].copy()
 # return controls to zero; final quench work
 for k in range(4):
  du=-old[k]
  if du: W += float(np.real(np.trace(rho@O[k])))*du
 P=float(np.real(np.trace(rho@rho)))
 E0=float(np.real(np.trace(rho0@H0))); Ef=float(np.real(np.trace(rho@H0)))
 dE=Ef-E0
 return (W,Q,dE,P,rho) if return_trace else (W,Q,dE,P)
def objective(c,lam,mu):
 W,Q,dE,P=simulate(c)
 v=max(0.,target-P)
 # Augmented penalty for inequality; update lambda outside SPSA
 return W + lam*v + .5*mu*v*v + 1e-6*dt*np.sum(make_u(c)**2), (W,Q,dE,P,v)
rng=np.random.default_rng(20261113)
bound=1.5; best=None; best_feas=None; records=[]
# multi-start SPSA with projected controls and augmented-Lagrangian updates
for seed in range(4):
 c=np.clip(rng.normal(0,.15,20),-bound,bound)
 lam=2.0; mu=200.0; local_best=None
 for it in range(1,101):
  ak=.035/(it+8)**.602; ck=.08/(it**.101)
  delta=rng.choice([-1.,1.],20)
  yp=objective(np.clip(c+ck*delta,-bound,bound),lam,mu)[0]
  ym=objective(np.clip(c-ck*delta,-bound,bound),lam,mu)[0]
  grad=(yp-ym)/(2*ck*delta)
  c=np.clip(c-ak*grad,-bound,bound)
  val,met=objective(c,lam,mu)
  if local_best is None or val<local_best[0]: local_best=(val,c.copy(),met)
  if it%10==0:
   lam=min(1e6,max(0.,lam+mu*met[4]))
   if met[4]>1e-4: mu=min(mu*2,1e6)
  if met[4]<=1e-5:
   if best_feas is None or met[0]<best_feas[0]:
    best_feas=(met[0],c.copy(),met)
 if best is None or local_best[0]<best[0]: best=local_best
# choose best feasible, otherwise lowest constraint violation (not a valid constrained optimum)
chosen=best_feas if best_feas is not None else best
W,Q,dE,P,rho=simulate(chosen[1],True)
res=dE-W-Q
result={
 'model':'two-qubit Lindblad amplitude damping; no explicit memory modes',
 'master_equation':'drho/dt=-i[H(t),rho]+D_gamma[rho]',
 'heat_convention':'Q positive when energy enters system; Q=integral Tr(H(t) D_gamma[rho]) dt',
 'work_convention':'W_ctrl from instantaneous control-field quenches, rho immediately before each quench; controls returned to zero',
 'gamma':gamma,'gamma_scan':scan,'target_purity':target,
 'optimization':{'method':'projected SPSA with augmented-Lagrangian inequality penalty','seeds':4,'iterations_per_seed':100,'control_bound':bound,'feasible_solution_found':best_feas is not None,'chosen_is_feasible':P>=target-1e-5,'W_ctrl':W,'Q_bath_to_system':Q,'delta_E_system':dE,'first_law_residual':res,'final_purity':P,'constraint_violation':max(0,target-P),'objective_with_regularizer':chosen[0]},
 'comparison_note':'No TDT inference: compare only after matching target, initial state, control channels/bounds, duration, endpoint Hamiltonian, and accounting conventions. The explicit-mode closed model and this reduced Lindblad model are not microscopically identical.',
}
with open(OUT+'/tdt_v5_markov_restricted_heat_result.json','w') as f: json.dump(result,f,indent=2)
with open(OUT+'/tdt_v5_markov_restricted_heat_result.txt','w') as f: f.write(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
