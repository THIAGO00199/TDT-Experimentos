#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply

OUT='/home/iknowyou/TDT_Experimentos'
T=18.; NT=180; dt=T/NT; K=9; BOUND=1.5; N=3; DIM=36
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
Hmem=(q1n+1.08*q2n+.82*m1n+1.18*m2n+.025*(2*q1n-Iq)@(2*q2n-Iq)
 +.32*(embed4(sp,0)@m1a+embed4(sp,0).getH()@m1a.getH())
 +.32*(embed4(sp,1)@m2a+embed4(sp,1).getH()@m2a.getH())).tocsr()
X=op([[0,1],[1,0]]); Y=op([[0,-1j],[1j,0]])
Om=[embed4(X,0),embed4(Y,0),embed4(X,1),embed4(Y,1)]
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
    z=psi.reshape(4,N*N); return z@z.conj().T
def sim_memory(c):
    u=controls(c); x=psi0.copy(); W=0.; old=np.zeros(4)
    for j in range(NT):
        for k in range(4):
            du=u[j,k]-old[k]
            if du: W+=float(np.real(np.vdot(x,Om[k]@x)))*du
        Hj=Hmem.copy()
        for k in range(4):
            if u[j,k]!=0: Hj=Hj+u[j,k]*Om[k]
        x=expm_multiply((-1j*dt)*Hj,x); old=u[j].copy()
    for k in range(4):
        du=-old[k]
        if du: W+=float(np.real(np.vdot(x,Om[k]@x)))*du
    rq=rho_qubits(x)
    return {'W_ctrl':W,'purity':float(np.real(np.trace(rq@rq))),'rho':rq,'state':x}
def spsa_memory(seed,niter=300,scale=.18):
    rng=np.random.default_rng(seed); best=None
    starts=[np.zeros(4*K)]+[np.clip(rng.normal(0,.35,4*K),-BOUND,BOUND) for _ in range(5)]
    for c in starts:
        val=-sim_memory(c)['purity']
        if best is None or val<best[0]: best=(val,c.copy())
    c=best[1].copy()
    for it in range(1,niter+1):
        ak=scale/(it+10)**.602; ck=.12/(it**.101); d=rng.choice([-1.,1.],4*K)
        yp=-sim_memory(np.clip(c+ck*d,-BOUND,BOUND))['purity']
        ym=-sim_memory(np.clip(c-ck*d,-BOUND,BOUND))['purity']
        grad=(yp-ym)/(2*ck*d); c=np.clip(c-ak*grad,-BOUND,BOUND)
        val=-sim_memory(c)['purity']
        if val<best[0]: best=(val,c.copy())
    return best[1]

# Markovian model, same qubit Hamiltonian and protocol.
I2=np.eye(2,dtype=complex); I4=np.eye(4,dtype=complex)
sp2=np.array([[0,1],[0,0]],complex); sm2=sp2.conj().T
X2=np.array([[0,1],[1,0]],complex); Y2=np.array([[0,-1j],[1j,0]],complex)
def emb2(a,s): return np.kron(a,I2) if s==0 else np.kron(I2,a)
n1=emb2(sm2@sp2,0); n2=emb2(sm2@sp2,1)
Hq=n1+1.08*n2+.025*(2*n1-I4)@(2*n2-I4)
Oq=[emb2(X2,0),emb2(Y2,0),emb2(X2,1),emb2(Y2,1)]
Cq=[emb2(sm2,0),emb2(sm2,1)]
bellq=np.array([1,0,0,1],complex)/np.sqrt(2); rho0=np.outer(bellq,bellq.conj())
def D(r):
    out=np.zeros((4,4),complex)
    for c0 in Cq:
        c=np.sqrt(.005)*c0; cd=c.conj().T; cdc=cd@c
        out+=c@r@cd-.5*(cdc@r+r@cdc)
    return out
def rhs(r,H): return -1j*(H@r-r@H)+D(r)
def rk4(r,H,h):
    k1=rhs(r,H); k2=rhs(r+.5*h*k1,H); k3=rhs(r+.5*h*k2,H); k4=rhs(r+h*k3,H)
    return r+(h/6)*(k1+2*k2+2*k3+k4)
def sim_markov(c):
    u=controls(c); r=rho0.copy(); W=0.; old=np.zeros(4)
    for j in range(NT):
        for k in range(4):
            du=u[j,k]-old[k]
            if du: W+=float(np.real(np.trace(r@Oq[k])))*du
        H=Hq.copy()
        for k in range(4): H+=u[j,k]*Oq[k]
        r=rk4(r,H,dt); old=u[j].copy()
    for k in range(4):
        du=-old[k]
        if du: W+=float(np.real(np.trace(r@Oq[k])))*du
    return {'W_ctrl':W,'purity':float(np.real(np.trace(r@r))),'rho':r}
def spsa_markov(seed,niter=300,scale=.18):
    rng=np.random.default_rng(seed); best=None
    starts=[np.zeros(4*K)]+[np.clip(rng.normal(0,.35,4*K),-BOUND,BOUND) for _ in range(5)]
    for c in starts:
        val=-sim_markov(c)['purity']
        if best is None or val<best[0]: best=(val,c.copy())
    c=best[1].copy()
    for it in range(1,niter+1):
        ak=scale/(it+10)**.602; ck=.12/(it**.101); d=rng.choice([-1.,1.],4*K)
        yp=-sim_markov(np.clip(c+ck*d,-BOUND,BOUND))['purity']
        ym=-sim_markov(np.clip(c-ck*d,-BOUND,BOUND))['purity']
        grad=(yp-ym)/(2*ck*d); c=np.clip(c-ak*grad,-BOUND,BOUND)
        val=-sim_markov(c)['purity']
        if val<best[0]: best=(val,c.copy())
    return best[1]

# Common subsystem reference: Hq, zero-temperature Gibbs ground state.
evals=np.linalg.eigvalsh(Hq); Eground=float(evals[0])
def state_metrics(rho):
    rho=(rho+rho.conj().T)/2
    E=float(np.real(np.trace(Hq@rho)))
    ev=np.maximum(np.linalg.eigvalsh(rho),1e-15)
    S=float(-np.sum(ev*np.log(ev)))
    B0=E-Eground
    return {'energy_Hq':E,'entropy_nats':S,'purity':float(np.real(np.trace(rho@rho))),
            'availability_Tref0':B0,'reference_ground_energy':Eground}
def pair_complex(z):
    a=np.asarray(z)
    if a.ndim==0: return [float(np.real(a)),float(np.imag(a))]
    return [pair_complex(v) for v in a]
def pack(label,model,seed,c,met):
    rho=met['rho']; return {'label':label,'model':model,'seed':seed,
      'purity':met['purity'],'W_ctrl':met['W_ctrl'],
      'control_parameters':np.asarray(c).tolist(),
      'rho_qubits_final_real_imag':pair_complex(rho),
      'exergy_reference':state_metrics(rho)}
t0=time.time(); results=[]
for seed in [0,14]:
    c=spsa_memory(20261115+seed)
    m=sim_memory(c)
    results.append(pack('nonmarkov_seed_'+str(seed),'closed two-mode explicit memory',seed,c,m))
    print('RECOVER_MEMORY',seed,m['purity'],flush=True)
c=spsa_markov(20261115)
m=sim_markov(c)
results.append(pack('markov_seed_0','two-qubit Lindblad amplitude damping, gamma=0.005',0,c,m))
print('RECOVER_MARKOV',m['purity'],flush=True)
out={'protocol':{'K':K,'T':T,'NT':NT,'dt':dt,'bound':BOUND,'iterations':300,'starts':6},
     'reference':'Same two-qubit Hq for both; T_ref=0, so availability is E-E_ground. This is not controller work.',
     'ground_energy_Hq':Eground,'results':results,'runtime_seconds':time.time()-t0,
     'complex_encoding':'Every complex matrix element is saved as [real, imaginary].',
     'warning':'Availability of the reduced qubit subsystem excludes explicit mode interaction energy/correlations; cross-model comparison is diagnostic, not a TDT proof.'}
base=OUT+'/tdt_v5_recovered_winners_exergy'
with open(base+'.json','w') as f: json.dump(out,f,indent=2,allow_nan=False)
with open(base+'.txt','w') as f: json.dump(out,f,indent=2,allow_nan=False)
print('RECOVERY_SAVED',base+'.json',flush=True)
print('RECOVERY_SUMMARY',json.dumps({'ground_energy_Hq':Eground,'results':[{'label':r['label'],'purity':r['purity'],'W_ctrl':r['W_ctrl'],'exergy_T0':r['exergy_reference']['availability_Tref0'],'energy':r['exergy_reference']['energy_Hq'],'entropy':r['exergy_reference']['entropy_nats']} for r in results]}),flush=True)
