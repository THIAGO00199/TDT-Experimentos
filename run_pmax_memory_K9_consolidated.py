#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply

OUT='/home/iknowyou/TDT_Experimentos'
T=18.0; NT=180; dt=T/NT; K=9; BOUND=1.5
N=3; DIM=36
def op(a): return csr_matrix(a,dtype=complex)
def embed4(a,site):
    dims=[2,2,N,N]; z=None
    for j,d in enumerate(dims):
        x=a if j==site else identity(d,format='csr',dtype=complex)
        z=x if z is None else kron(z,x,format='csr')
    return z
sp=op([[0,1],[0,0]]); sm=sp.getH(); num=sm@sp
ann=csr_matrix(np.diag(np.sqrt(np.arange(1,N)),1).astype(complex)); adag=ann.getH(); nm=adag@ann
Iq=identity(DIM,format='csr',dtype=complex)
q1n,q2n=embed4(num,0),embed4(num,1)
m1a,m2a=embed4(ann,2),embed4(ann,3)
m1n,m2n=embed4(nm,2),embed4(nm,3)
H=(1.0*q1n+1.08*q2n+.82*m1n+1.18*m2n+.025*(2*q1n-Iq)@(2*q2n-Iq)
   +.32*(embed4(sp,0)@m1a+embed4(sp,0).getH()@m1a.getH())
   +.32*(embed4(sp,1)@m2a+embed4(sp,1).getH()@m2a.getH())).tocsr()
X=op([[0,1],[1,0]]); Y=op([[0,-1j],[1j,0]])
O=[embed4(X,0),embed4(Y,0),embed4(X,1),embed4(Y,1)]
bell=np.zeros(4,complex); bell[[0,3]]=1/np.sqrt(2)
vac=np.zeros(N*N,complex); vac[0]=1
psi0=np.kron(bell,vac)
grid=np.arange(NT); knots=np.linspace(0,NT-1,K)
def controls(c):
    cc=np.asarray(c).reshape(4,K); u=np.zeros((NT,4))
    for k in range(4): u[:,k]=np.interp(grid,knots,cc[k])
    u[0]=0.; u[-1]=0.
    return u
def rho_qubits(psi):
    z=psi.reshape(4,N*N)
    return z@z.conj().T
def sim_memory(c,save=False):
    u=controls(c); x=psi0.copy(); W=0.; old=np.zeros(4)
    for j in range(NT):
        for k in range(4):
            du=u[j,k]-old[k]
            if du: W+=float(np.real(np.vdot(x,O[k]@x)))*du
        Hj=H.copy()
        for k in range(4):
            if u[j,k]!=0: Hj=Hj+u[j,k]*O[k]
        x=expm_multiply((-1j*dt)*Hj,x)
        old=u[j].copy()
    for k in range(4):
        du=-old[k]
        if du: W+=float(np.real(np.vdot(x,O[k]@x)))*du
    rq=rho_qubits(x); P=float(np.real(np.trace(rq@rq)))
    E0=float(np.real(np.vdot(psi0,H@psi0)))
    Ef=float(np.real(np.vdot(x,H@x)))
    S=float(-np.sum(np.maximum(np.linalg.eigvalsh((rq+rq.conj().T)/2),1e-15)*
                       np.log(np.maximum(np.linalg.eigvalsh((rq+rq.conj().T)/2),1e-15))))
    met={'W_ctrl':W,'Q_explicit_bath':0.0,'delta_E_total':Ef-E0,'purity':P,'entropy_nats':S}
    if save: met['rho_qubits_final']=rq.tolist()
    return met
def spsa(seed,niter=300,scale=.18):
    rng=np.random.default_rng(seed); best=None
    starts=[np.zeros(4*K)]+[np.clip(rng.normal(0,.35,4*K),-BOUND,BOUND) for _ in range(5)]
    for c in starts:
        val=-sim_memory(c)['purity']
        if best is None or val<best[0]: best=(val,c.copy())
    c=best[1].copy()
    for it in range(1,niter+1):
        ak=scale/(it+10)**.602; ck=.12/(it**.101)
        d=rng.choice([-1.,1.],4*K)
        yp=-sim_memory(np.clip(c+ck*d,-BOUND,BOUND))['purity']
        ym=-sim_memory(np.clip(c-ck*d,-BOUND,BOUND))['purity']
        grad=(yp-ym)/(2*ck*d)
        c=np.clip(c-ak*grad,-BOUND,BOUND)
        val=-sim_memory(c)['purity']
        if val<best[0]: best=(val,c.copy())
    return best
t0=time.time(); records=[]
for seed in range(20):
    _,c=spsa(20261115+seed,300,.18)
    met=sim_memory(c,save=True)
    rec={'seed':seed,**met,'control_parameters':c.tolist()}
    records.append(rec)
    print('MEMORY_PMAX_SEED',seed,'P',met['purity'],'W',met['W_ctrl'],flush=True)
best=max(records,key=lambda r:r['purity'])
vals=np.array([r['purity'] for r in records])
result={
 'model':'two qubits + two explicit truncated N=3 oscillator modes; closed unitary dynamics',
 'protocol':{'K':K,'T':T,'NT':NT,'dt':dt,'control_bound':BOUND,'control_channels':'local X/Y on each qubit','seeds':20,'iterations_per_seed':300,'initial_candidates_per_seed':6,'seed_base':20261115,'objective':'maximize final reduced two-qubit purity directly'},
 'hamiltonian_parameters':{'wq':[1.0,1.08],'wa':[0.82,1.18],'g':[0.32,0.32],'Jzz':0.025,'mode_cutoff':N,'hilbert_dimension':DIM},
 'seeds':records,
 'best_found':best,
 'mean_purity':float(np.mean(vals)),
 'std_purity_sample':float(np.std(vals,ddof=1)),
 'zero_control_or_unoptimized_count':int(sum(abs(r['W_ctrl'])<1e-12 and abs(r['delta_E_total'])<1e-10 for r in records)),
 'certified_upper_bound':False,
 'notes':['Same K=9, T, NT, control bounds, local X/Y channels, six starts, 20 seed values and 300 SPSA iterations as the completed Markovian run.','The explicit-mode model is closed and unitary; Q=0 applies only to the full explicitly modelled system.','Purity is for the reduced two-qubit subsystem.','Best-found SPSA value is not a global optimum or a certified physical ceiling.','Final reduced density matrices and control vectors are saved for later matched-reference availability/exergy analysis.']
}
base=OUT+'/tdt_v5_pmax_memory_K9_20seeds_300iter'
with open(base+'.json','w') as f: json.dump(result,f,indent=2)
with open(base+'.txt','w') as f: json.dump(result,f,indent=2)
print('MEMORY_CONSOLIDATED_RESULT',json.dumps({k:v for k,v in result.items() if k!='seeds'},separators=(',',':')),flush=True)
print('MEMORY_RUNTIME',time.time()-t0,flush=True)
