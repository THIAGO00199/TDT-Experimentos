#!/usr/bin/env python3
"""Non-Markovian two-qubit/two-mode entropy-control experiment.
Explicit finite environment; closed unitary dynamics, hbar=1.
Control: five temporal knots per local X/Y channel on each qubit.
Objective: minimize final two-qubit von Neumann entropy + small pulse penalty.
"""
import json, time
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply

N=3; DT=4*N*N
wq1,wq2=1.0,1.08; wa1,wa2=0.82,1.18
g1,g2=0.32,0.32; Jzz=0.025
T=18.0; NT=180; dt=T/NT
rng=np.random.default_rng(20261114)

def op2(x): return csr_matrix(x,dtype=complex)
def embed(op,site):
    dims=[2,2,N,N]; out=None
    for k,d in enumerate(dims):
        x=op if k==site else identity(d,format='csr',dtype=complex)
        out=x if out is None else kron(out,x,format='csr')
    return out
I=identity(DT,format='csr',dtype=complex)
sp=op2([[0,1],[0,0]]); sm=sp.getH(); num=sm@sp
ann=csr_matrix(np.diag(np.sqrt(np.arange(1,N)),1).astype(complex)); adag=ann.getH(); n=adag@ann
q1n,q2n=embed(num,0),embed(num,1)
q1p,q2p=embed(sp,0),embed(sp,1)
m1a,m2a=embed(ann,2),embed(ann,3)
m1n,m2n=embed(n,2),embed(n,3)
H=(wq1*q1n+wq2*q2n+wa1*m1n+wa2*m2n+Jzz*(2*q1n-I)@(2*q2n-I)
   +g1*(q1p@m1a+q1p.getH()@m1a.getH())+g2*(q2p@m2a+q2p.getH()@m2a.getH())).tocsr()
X=op2([[0,1],[1,0]]); Y=op2([[0,-1j],[1j,0]])
DR=[embed(X,0),embed(Y,0),embed(X,1),embed(Y,1)]
bell=np.zeros(4,complex); bell[0]=bell[3]=1/np.sqrt(2)
vac=np.zeros(N*N,complex); vac[0]=1
psi0=np.kron(bell,vac)

def rho_q(psi):
    z=psi.reshape(4,N*N); return z@z.conj().T
def entropy(rho):
    ev=np.linalg.eigvalsh((rho+rho.conj().T)/2); ev=np.clip(ev,0,1)
    return float(-np.sum(ev*np.log(np.maximum(ev,1e-15))))
def purity(rho): return float(np.real(np.trace(rho@rho)))
def concurrence(rho):
    sy=np.array([[0,-1j],[1j,0]]); YY=np.kron(sy,sy); R=rho@YY@rho.conj()@YY
    ev=np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)),0)))[::-1]
    return float(max(0.,ev[0]-ev[1]-ev[2]-ev[3]))
def evolve(psi,us=None):
    x=psi.copy()
    for j in range(NT):
        h=H
        if us is not None:
            for k in range(4):
                if us[j,k]!=0: h=h+us[j,k]*DR[k]
        x=expm_multiply((-1j*dt)*h,x)
    return x
# Free BLP witness for a standard orthogonal pair |+0>, |-0>.
pplus=np.zeros(4,complex); pplus[0]=pplus[2]=1/np.sqrt(2)
pminus=np.zeros(4,complex); pminus[0]=1/np.sqrt(2); pminus[2]=-1/np.sqrt(2)
pa=np.kron(pplus,vac); pb=np.kron(pminus,vac)
D=[]
for j in range(NT+1):
    if j:
        pa=expm_multiply((-1j*dt)*H,pa); pb=expm_multiply((-1j*dt)*H,pb)
    D.append(float(np.sum(np.abs(np.linalg.eigvalsh(rho_q(pa)-rho_q(pb))))/2))
D=np.array(D); dD=np.diff(D); pos=np.where(dD>1e-10)[0]
if len(pos):
    s=int(pos[0]); e=int(pos[-1]+1)
else: s,e=0,NT
# Widen to include all intervals with positive derivative, including gaps.
s=max(0,s); e=min(NT,e)
# Five knots per channel, interpolated across the entire detected backflow span.
KNOTS=5; ncoef=4*KNOTS; bound=2.0
tk=np.linspace(s,e,KNOTS)

def make_controls(c):
    cc=c.reshape(4,KNOTS); us=np.zeros((NT,4)); grid=np.arange(NT)
    for k in range(4): us[:,k]=np.interp(grid,tk,cc[k],left=0.,right=0.)
    # ensure control vanishes at the exact outer boundaries
    us[:s]=0.; us[e:]=0.
    return us
# Objective is final subsystem entropy. Effort penalty only regularizes, not thermodynamic work.
lam=1e-4
def objective(c,details=False):
    us=make_controls(c); x=evolve(psi0,us); rq=rho_q(x)
    S=entropy(rq); P=purity(rq); effort=float(dt*np.sum(us*us))
    val=S+lam*effort
    return (val,S,P,effort,x) if details else val
basepsi=evolve(psi0); baserho=rho_q(basepsi)
baseS=entropy(baserho); baseP=purity(baserho); baseC=concurrence(baserho)
# SPSA
c=np.zeros(ncoef); best=c.copy(); bestv=objective(c)
# Small deterministic multistart so SPSA is not trapped at the zero-control stationary point.
random_seeds=[]
for trial in range(1,21):
    candidate=np.clip(rng.normal(0.0,0.45,size=ncoef),-bound,bound)
    candidate_value=objective(candidate)
    random_seeds.append(float(candidate_value))
    if candidate_value<bestv:
        bestv=candidate_value; best=candidate.copy()
c=best.copy()
a_spsa=0.16; c_spsa=0.20; A=12; alpha=.602; gamma=.101
history=[]; t0=time.time()
for it in range(1,121):
    ak=a_spsa/(it+A)**alpha; ck=c_spsa/(it**gamma)
    delta=rng.choice([-1.,1.],size=ncoef)
    yp=objective(np.clip(c+ck*delta,-bound,bound)); ym=objective(np.clip(c-ck*delta,-bound,bound))
    grad=(yp-ym)/(2*ck*delta)
    c=np.clip(c-ak*grad,-bound,bound)
    val=objective(c)
    if val<bestv: bestv=val; best=c.copy()
    if it%10==0: history.append({"iteration":it,"objective":float(val),"best":float(bestv)})
optobj,optS,optP,effort,optpsi=objective(best,True)
# Midpoint discrete work estimate W = integral <O_k> du_k.
us=make_controls(best); x=psi0.copy(); work=0.0
for j in range(NT):
    h=H
    for k in range(4):
        if us[j,k]!=0: h=h+us[j,k]*DR[k]
    xm=expm_multiply((-0.5j*dt)*h,x)
    for k in range(4):
        prev=us[j-1,k] if j else 0.0
        du=us[j,k]-prev
        work+=float(np.real(np.vdot(xm,DR[k]@xm)))*du
    x=expm_multiply((-1j*dt)*h,x)
# Verify work against total energy change for cyclic external drive (numerical discretization residual).
E0=float(np.real(np.vdot(psi0,H@psi0))); Ef=float(np.real(np.vdot(x,H@x)))
result={
 "model":"two qubits plus two explicit truncated harmonic modes; closed unitary dynamics",
 "parameters":{"mode_cutoff":N,"hilbert_dimension":DT,"wq":[wq1,wq2],"wa":[wa1,wa2],"g":[g1,g2],"Jzz":Jzz,"T":T,"NT":NT,"dt":dt},
 "backflow":{"BLP_pair":"|+0> vs |-0>, both modes vacuum","D_initial":float(D[0]),"D_final":float(D[-1]),"Pi_BLP_positive_variation":float(np.maximum(dD,0).sum()),"first_positive_interval_time":float(s*dt),"last_positive_interval_time":float(e*dt),"positive_derivative_steps":int(len(pos))},
 "baseline":{"S_qubits_nats":baseS,"purity":baseP,"concurrence":baseC,"energy_total":E0},
 "control":{"objective":"minimize final von Neumann entropy of reduced two-qubit state + 1e-4 integral(sum(u_k^2)dt)","method":"SPSA 120 iterations; 5 temporal knots per each of 4 local X/Y channels; control spans all positive-derivative backflow interval","parameter_count":ncoef,"control_window_steps":[s,e],"history":history,"random_multistart_20_best_objective":float(min(random_seeds)),"optimized_S_qubits_nats":optS,"optimized_purity":optP,"optimized_concurrence":concurrence(rho_q(optpsi)),"entropy_reduction":baseS-optS,"purity_gain":optP-baseP,"control_effort_proxy_u2":effort,"work_midpoint_estimate":work,"energy_change_total":Ef-E0,"work_energy_balance_residual":work-(Ef-E0),"runtime_seconds":time.time()-t0},
 "thermodynamic_bound_note":"No numerical quantum-thermodynamic minimum-work bound is asserted: this is a finite closed unitary system with vacuum modes, not an isothermal reservoir. For a specified closed-system protocol, work equals the total energy change; an isothermal bound W>=Delta F would require a thermal bath and a defined initial/final global state and Hamiltonian. The entropy objective alone does not define a unique reversible work target.",
 "caveats":["The BLP quantity is a one-pair witness, not TDT debt.","Local reduced entropy minimization can restore purity without restoring the initial Bell state.","The work estimate is a midpoint discretization and its energy-balance residual is reported; it is not a universal actuator-cost measure.","A fair TDT test requires matched target constraints and a conventional optimal-control lower bound under the same Hamiltonian and accessible controls."]
}
out='/home/iknowyou/TDT_Experimentos/tdt_quantum_nonmarkovian_entropy_control_result.json'
with open(out,'w') as f: json.dump(result,f,indent=2)
with open('/home/iknowyou/TDT_Experimentos/tdt_quantum_nonmarkovian_entropy_control_result.txt','w') as f: f.write(json.dumps(result,indent=2))
print('BACKFLOW',json.dumps(result['backflow']))
print('BASELINE',json.dumps(result['baseline']))
print('CONTROL',json.dumps(result['control']))
print('BOUND',result['thermodynamic_bound_note'])
print('SAVED',out)
