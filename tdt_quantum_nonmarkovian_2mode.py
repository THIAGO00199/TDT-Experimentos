#!/usr/bin/env python3
"""Two-qubit + two explicit oscillator modes: non-Markovian backflow pilot.
Closed-system unitary benchmark; no Lindblad reservoir. All energies hbar=1.
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import eigvalsh

N=3; DQ=4; DT=DQ*N*N
wq1,wq2=1.0,1.08
wa1,wa2=0.82,1.18
g1,g2=0.32,0.32
Jzz=0.025
T=18.0; NT=180; dt=T/NT
rng=np.random.default_rng(20261113)
I2=identity(2,format='csr',dtype=complex); IN=identity(N,format='csr',dtype=complex)
def op2(a): return csr_matrix(a,dtype=complex)
sz=op2([[1,0],[0,-1]])
sp=op2([[0,1],[0,0]])
sm=sp.getH()
num=sm@sp
ann=np.diag(np.sqrt(np.arange(1,N)),1).astype(complex)
a=csr_matrix(ann); adag=a.getH(); n=a.getH()@a
# Tensor order q1,q2,m1,m2
ops=[]
for local in [I2,I2,IN,IN]: pass
def embed(op,site):
    dims=[2,2,N,N]; out=None
    for k,d in enumerate(dims):
        x=op if k==site else identity(d,format='csr',dtype=complex)
        out=x if out is None else kron(out,x,format='csr')
    return out
q1n,q2n=embed(num,0),embed(num,1)
q1p,q2p=embed(sp,0),embed(sp,1)
m1a,m2a=embed(a,2),embed(a,3)
m1n,m2n=embed(n,2),embed(n,3)
H=(wq1*q1n+wq2*q2n+wa1*m1n+wa2*m2n+Jzz*(2*q1n-identity(DT,format='csr'))@(2*q2n-identity(DT,format='csr'))
   +g1*(q1p@m1a+q1p.getH()@m1a.getH())+g2*(q2p@m2a+q2p.getH()@m2a.getH())).tocsr()
# local qubit drive operators
X1=embed(op2([[0,1],[1,0]]),0); Y1=embed(op2([[0,-1j],[1j,0]]),0)
X2=embed(op2([[0,1],[1,0]]),1); Y2=embed(op2([[0,-1j],[1j,0]]),1)
DR=[X1,Y1,X2,Y2]
# initial Bell pair and oscillator vacuum
bell=np.zeros(4,complex); bell[0]=bell[3]=1/np.sqrt(2)
vac=np.zeros(N*N,complex); vac[0]=1
psi0=np.kron(bell,vac)
# orthogonal pair for BLP trace-distance witness |00>, |11>
pplus=np.zeros(4,complex); pplus[0]=pplus[2]=1/np.sqrt(2)  # |+0>
pminus=np.zeros(4,complex); pminus[0]=1/np.sqrt(2); pminus[2]=-1/np.sqrt(2)  # |-0>
phi0=np.kron(pplus,vac); phi1=np.kron(pminus,vac)
# partial trace oscillator from pure state; reshape q1,q2,m1,m2
def rho_q(psi):
    z=psi.reshape(4,N*N); return z@z.conj().T
def trace_distance(r1,r2):
    return float(np.sum(np.abs(np.linalg.eigvalsh(r1-r2)))/2)
def concurrence(rho):
    sy=np.array([[0,-1j],[1j,0]])
    YY=np.kron(sy,sy); R=rho@YY@rho.conj()@YY
    ev=np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)),0)))[::-1]
    return float(max(0.0,ev[0]-ev[1]-ev[2]-ev[3]))
def entropy(rho):
    ev=np.linalg.eigvalsh((rho+rho.conj().T)/2); ev=np.clip(ev,1e-15,1)
    return float(-np.sum(ev*np.log(ev)))
def propagate(psi, controls=None, start=0, stop=NT):
    x=psi.copy()
    for j in range(start,stop):
        h=H
        if controls is not None:
            u=controls[j]
            for k in range(4):
                if abs(u[k])>0: h=h+u[k]*DR[k]
        x=expm_multiply((-1j*dt)*h,x)
    return x
# Free evolution time series
bell_states=[]; dists=[]; conc=[]; ent=[]
pa,pb=phi0.copy(),phi1.copy()
for j in range(NT+1):
    if j>0:
        pa=expm_multiply((-1j*dt)*H,pa); pb=expm_multiply((-1j*dt)*H,pb)
    dists.append(trace_distance(rho_q(pa),rho_q(pb)))
    if j%1==0:
        psi=propagate(psi0,stop=j) if j<=0 else None
        # Bell trajectory propagated incrementally below instead
    if j==0: psib=psi0.copy()
    elif True: psib=expm_multiply((-1j*dt)*H,psib)
    rb=rho_q(psib); conc.append(concurrence(rb)); ent.append(entropy(rb))
dists=np.array(dists); back=np.maximum(np.diff(dists),0); Pi=float(np.sum(back))
# pick strongest positive backflow interval, control window centered there
idx=int(np.argmax(back))+1 if np.max(back)>1e-10 else int(np.argmax(dists))
center_t=idx*dt; win=max(12,min(30,NT//3)); s=max(0,idx-win//2); e=min(NT,s+win); s=max(0,e-win)
# SPSA optimizes controls only within selected backflow window, 4 channels x 3 smooth basis coeffs
ncoef=12; bound=1.2
xgrid=np.linspace(-1,1,e-s)
P=np.column_stack([np.ones_like(xgrid),xgrid,0.5*(3*xgrid*xgrid-1)])
shape=np.sin(np.pi*(np.arange(e-s)+0.5)/(e-s))[:,None]
def make_controls(c):
    us=np.zeros((NT,4)); cc=c.reshape(4,3)
    us[s:e]=shape*(P@cc.T)
    return us
def objective(c):
    us=make_controls(c); x=psi0.copy()
    for j in range(NT):
        h=H
        if np.any(us[j]):
            for k in range(4): h=h+us[j,k]*DR[k]
        x=expm_multiply((-1j*dt)*h,x)
    fid=float(np.real(np.vdot(bell,rho_q(x)@bell)))
    effort=float(dt*np.sum(us*us))
    return 1-fid+0.002*effort, fid, effort
# Baseline is exact free evolution at final time
basepsi=propagate(psi0)
basefid=float(np.real(np.vdot(bell,rho_q(basepsi)@bell)))
baseobj=1-basefid
c=np.zeros(ncoef); best=c.copy(); bestv=objective(c)[0]
a_spsa=0.08; c_spsa=0.12; A=8; alpha=0.602; gamma=0.101
log=[]
for k in range(1,61):
    ak=a_spsa/(k+A)**alpha; ck=c_spsa/(k**gamma)
    delta=rng.choice([-1.0,1.0],size=ncoef)
    yp=objective(np.clip(c+ck*delta,-bound,bound))[0]
    ym=objective(np.clip(c-ck*delta,-bound,bound))[0]
    grad=(yp-ym)/(2*ck*delta)
    c=np.clip(c-ak*grad,-bound,bound)
    val=objective(c)[0]
    if val<bestv: bestv=val; best=c.copy()
    if k%10==0: log.append({"iteration":k,"objective":val,"best":bestv})
optobj,optfid,effort=objective(best)
# Compute work using midpoint expectation and field increments; report sign convention W=integral <O> du
us=make_controls(best); x=psi0.copy(); work=0.0
for j in range(NT):
    h=H
    if np.any(us[j]):
        for k in range(4):
            if us[j,k]: h=h+us[j,k]*DR[k]
    # symmetric midpoint estimate for explicit drive work
    xm=expm_multiply((-0.5j*dt)*h,x)
    for k in range(4):
        if j>0: du=us[j,k]-us[j-1,k]
        else: du=us[j,k]
        work+=float(np.real(np.vdot(xm,DR[k]@xm)))*du
    x=expm_multiply((-1j*dt)*h,x)
opt_rho=rho_q(x)
result={"model":"2 qubits + 2 explicit truncated harmonic modes, closed unitary dynamics","parameters":{"N_mode":N,"dimension":DT,"wq":[wq1,wq2],"wa":[wa1,wa2],"g":[g1,g2],"Jzz":Jzz,"T":T,"NT":NT,"dt":dt},"blp_witness":{"initial_pair":"|+0> and |-0> with both modes in vacuum","D_initial":dists[0],"D_final":dists[-1],"D_max":float(np.max(dists)),"Pi_positive_variation_sum":Pi,"strongest_backflow_time":center_t,"strongest_increment":float(np.max(back)),"backflow_window_steps":[s,e],"D_series":dists.tolist()},"bell_free":{"fidelity_initial_target_at_T":basefid,"concurrence_final":conc[-1],"S_qubits_final_nats":ent[-1]},"control":{"method":"SPSA 60 iterations, 12 coefficients, four local X/Y drives; active only in selected backflow window","history":log,"baseline_fidelity":basefid,"optimized_fidelity":optfid,"baseline_infidelity":1-basefid,"optimized_infidelity":1-optfid,"control_effort_integral_u2":effort,"work_estimate_midpoint":work,"final_concurrence":concurrence(opt_rho),"final_S_qubits_nats":entropy(opt_rho)},"caveats":["Pi is the BLP positive variation for one chosen pair of initial states, not a new TDT quantity yet.","The oscillator modes are an explicit finite memory environment; the total evolution is unitary and there is no external Lindblad bath.","The SPSA control objective maximizes the Bell-state fidelity of the reduced two-qubit state; the oscillator modes are not required to return to vacuum.","Work estimate uses a discrete midpoint convention and must be checked against a fully specified actuator Hamiltonian before thermodynamic interpretation."]}
out=Path(__file__).resolve().parent / 'tdt_quantum_nonmarkovian_2mode_strong_result.json'
with open(out,'w') as f: json.dump(result,f,indent=2)
with open(Path(__file__).resolve().parent / 'tdt_quantum_nonmarkovian_2mode_strong_result.txt','w') as f:
    f.write(json.dumps(result,indent=2))
print('BACKFLOW',json.dumps(result['blp_witness']|{'D_series':'omitted from console'}))
print('BELL_FREE',json.dumps(result['bell_free']))
print('CONTROL',json.dumps(result['control']))
print('SAVED',out)
