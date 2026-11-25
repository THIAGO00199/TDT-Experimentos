#!/usr/bin/env python3
"""Quantum TDT pilot: two qubits + one truncated damped harmonic mode."""
import json, math, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply

OUT=Path("/home/iknowyou/TDT_Experimentos")
NQ=2
NOSC=6
DT=0.04
NSTEPS=200
T=DT*NSTEPS
HBAR=1.0
WQ1=1.0
WQ2=1.13
WA=1.0
G=0.38
KAPPA=0.18
NBAR=0.12
JZZ=0.04

def sp(a,b): return kron(csr_matrix(a),csr_matrix(b),format="csr")
I2=identity(2,format="csr",dtype=complex)
Iosc=identity(NOSC,format="csr",dtype=complex)
X=csr_matrix(np.array([[0,1],[1,0]],complex))
Y=csr_matrix(np.array([[0,-1j],[1j,0]],complex))
Z=csr_matrix(np.array([[1,0],[0,-1]],complex))
sm=csr_matrix(np.array([[0,1],[0,0]],complex))
sz1=sp(Z,I2); sz2=sp(I2,Z)
sx1=sp(X,I2); sx2=sp(I2,X)
sy1=sp(Y,I2); sy2=sp(I2,Y)
sm1=sp(sm,I2); sm2=sp(I2,sm)
Iq=identity(4,format="csr",dtype=complex)
a=np.zeros((NOSC,NOSC),complex)
for n in range(1,NOSC): a[n-1,n]=math.sqrt(n)
a=csr_matrix(a); adag=a.getH(); n_op=adag@a
# Tensor ordering: qubit 1, qubit 2, oscillator.
def with_osc(op): return kron(op,Iosc,format="csr")
def with_q(op): return kron(Iq,op,format="csr")
Sx1,Sx2,Sy1,Sy2=map(with_osc,(sx1,sx2,sy1,sy2))
Sm2=with_osc(sm2)
A=with_q(a); Ad=A.getH()
Hq=0.5*WQ1*with_osc(sz1)+0.5*WQ2*with_osc(sz2)+JZZ*with_osc(sz1@sz2)
Ha=WA*with_q(n_op)
Hint=G*(Sm2@Ad+Sm2.getH()@A)
H0=(Hq+Ha+Hint).tocsr()
D=4*NOSC
Id=identity(D,format="csr",dtype=complex)
# Collapse operators: oscillator thermal bath. Rates use standard nbar convention.
c_ops=[math.sqrt(KAPPA*(NBAR+1))*A, math.sqrt(KAPPA*NBAR)*Ad]
def dissipator(c):
    cd_c=c.getH()@c
    return kron(c.conjugate(),c,format="csr") - 0.5*kron(Id,cd_c,format="csr") - 0.5*kron(cd_c.transpose(),Id,format="csr")
L_diss=sum((dissipator(c) for c in c_ops),csr_matrix((D*D,D*D),dtype=complex))
def liouvillian(H):
    return (-1j*(kron(Id,H,format="csr")-kron(H.transpose(),Id,format="csr"))+L_diss).tocsr()
L0=liouvillian(H0)
# Bell state (|00>+|11>)/sqrt(2), oscillator thermal state.
bell=np.zeros(4,complex); bell[0]=bell[3]=1/math.sqrt(2)
rhoq=np.outer(bell,bell.conj())
p_osc=np.array([(NBAR**n)/(1+NBAR)**(n+1) for n in range(NOSC)])
p_osc/=p_osc.sum()
rhoosc=np.diag(p_osc)
rho=np.kron(rhoq,rhoosc).astype(complex)
rho_vec=rho.reshape(-1,order="F")
# Partial traces by tensor reshaping (q1,q2,osc; q1',q2',osc').
def partial_trace(r, keep):
    t=r.reshape(2,2,NOSC,2,2,NOSC)
    if keep==[0,1]:
        return np.einsum("ijkabl->ijab",t) if False else np.einsum("ijkabk->ijab",t).reshape(4,4)
    if keep==[2]:
        return np.einsum("ijkijc->kc",t)
    raise ValueError("unsupported subsystem")
def entropy(r):
    vals=np.linalg.eigvalsh((r+r.conj().T)/2)
    vals=np.clip(vals,0,1)
    vals=vals[vals>1e-14]
    return float(-np.sum(vals*np.log(vals)))
def negativity(r,dA=4,dB=NOSC):
    t=r.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(dA*dB,dA*dB)
    ev=np.linalg.eigvalsh((t+t.conj().T)/2)
    return float((np.sum(np.abs(ev))-1)/2)
def concurrence(r):
    sy=np.array([[0,-1j],[1j,0]],complex)
    YY=np.kron(sy,sy)
    R=r@YY@r.conj()@YY
    vals=np.linalg.eigvals(R)
    roots=np.sort(np.sqrt(np.clip(np.real(vals),0,None)))[::-1]
    return float(max(0.0,roots[0]-sum(roots[1:])))
def metrics(rv):
    rr=rv.reshape((D,D),order="F")
    rq=partial_trace(rr,[0,1])
    ra=partial_trace(rr,[2])
    sq=entropy(rq); sa=entropy(ra); st=entropy(rr)
    return {"S_qubits":sq,"S_osc":sa,"S_total":st,
            "I_qubits_osc":sq+sa-st,
            "negativity_qubits_osc":negativity(rr),
            "concurrence_qubits":concurrence(rq),
            "purity_qubits":float(np.real(np.trace(rq@rq))),
            "purity_global":float(np.real(np.trace(rr@rr))),
            "trace_error":float(abs(np.trace(rr)-1))}
def energy(rv,H):
    rr=rv.reshape((D,D),order="F")
    return float(np.real(np.trace(rr@H.toarray())))
def propagate(rv,H,dt):
    return expm_multiply(liouvillian(H)*dt,rv)

# Open-loop coherent control: 4 channels (X/Y on each qubit), 3 smooth Legendre modes each.
from scipy.special import eval_legendre
from scipy.optimize import minimize
CTRL_OPS=[Sx1,Sy1,Sx2,Sy2]
NMODES=3
NCTRL=len(CTRL_OPS)*NMODES
BELL_PROJ=np.outer(bell,bell.conj())
def u_at(t,theta):
    x=2*t/T-1
    env=math.sin(math.pi*t/T)
    c=theta.reshape(len(CTRL_OPS),NMODES)
    return env*np.array([sum(c[j,k]*eval_legendre(k,x) for k in range(NMODES)) for j in range(len(CTRL_OPS))])
def simulate_control(theta, keep_series=False):
    rv=rho.reshape(-1,order="F").copy()
    old=np.zeros(len(CTRL_OPS))
    work=0.0
    effort=0.0
    for n in range(NSTEPS):
        tm=(n+0.5)*DT
        u=u_at(tm,theta)
        rr=rv.reshape((D,D),order="F")
        for j,op in enumerate(CTRL_OPS):
            du=u[j]-old[j]
            if abs(du)>0:
                work+=float(np.real(np.trace(rr@(op.toarray()))))*du
        H=H0.copy()
        for uj,op in zip(u,CTRL_OPS):
            if abs(uj)>0: H=H+uj*op
        rv=expm_multiply(liouvillian(H)*DT,rv)
        effort+=DT*float(np.dot(u,u))
        old=u
    rr=rv.reshape((D,D),order="F")
    for j,op in enumerate(CTRL_OPS):
        work+=float(np.real(np.trace(rr@op.toarray())))*(0.0-old[j])
    rq=partial_trace(rr,[0,1])
    fidelity=float(np.real(np.trace(rq@BELL_PROJ)))
    m=metrics(rv)
    return {"fidelity":fidelity,"infidelity":1-fidelity,"work_control":work,
            "control_effort_integral_u2":effort,"metrics":m,"rho_final":rv if keep_series else None}
def objective(theta):
    z=simulate_control(theta)
    return z["infidelity"]+0.001*z["control_effort_integral_u2"]
theta=np.zeros(NCTRL)
baseline=simulate_control(theta)
best_theta=theta.copy()
best_obj=objective(theta)
rng=np.random.default_rng(20261113)
history=[]
# SPSA; identical deterministic solver and fixed model for all evaluations.
for k in range(60):
    ck=0.10/((k+1)**0.101)
    ak=0.12/((k+20)**0.602)
    delta=rng.choice([-1.0,1.0],size=NCTRL)
    yp=objective(theta+ck*delta)
    ym=objective(theta-ck*delta)
    grad=((yp-ym)/(2*ck))*delta
    theta=np.clip(theta-ak*grad,-1.5,1.5)
    val=objective(theta)
    if val<best_obj:
        best_obj=val;best_theta=theta.copy()
    history.append({"iteration":k+1,"objective":val,"best_objective":best_obj,
                    "spsa_gradient_norm":float(np.linalg.norm(grad))})
    if (k+1)%10==0: print("SPSA",k+1,"objective",val,"best",best_obj,flush=True)
optimized=simulate_control(best_theta)
result={
 "model":"2 qubits + one truncated oscillator + thermal Lindblad reservoir",
 "optimizer":"SPSA, 60 iterations, 12 smooth open-loop pulse coefficients",
 "control_channels":["X_q1","Y_q1","X_q2","Y_q2"],
 "basis":"sin(pi*t/T) times Legendre modes k=0,1,2",
 "objective":"1 - Bell-state fidelity of reduced two-qubit state + 0.001*integral(sum_j u_j(t)^2 dt)",
 "baseline":{k:v for k,v in baseline.items() if k!="rho_final"},
 "optimized":{k:v for k,v in optimized.items() if k!="rho_final"},
 "theta":best_theta.tolist(),"history":history,
 "parameters":{"iterations":60,"coefficient_bounds":[-1.5,1.5],"spsa_a":0.12,"spsa_c":0.10,
               "spsa_A":20,"spsa_alpha":0.602,"spsa_gamma":0.101,
               "T":T,"dt":DT,"steps":NSTEPS,"g":G,"kappa":KAPPA,"nbar":NBAR},
 "warning":"This is an open-loop pulse benchmark. The objective regularizer integral(u^2) is a control-effort proxy, not thermodynamic work. Signed work is separately calculated as sum Tr(rho O_j) delta u_j."
}
p=OUT/"tdt_quantum_lindblad_control_spsa.json"
p.write_text(json.dumps(result,indent=2))
(OUT/"tdt_quantum_lindblad_control_spsa.txt").write_text(json.dumps(result,indent=2)+"\n")
print("BASELINE",json.dumps(result["baseline"]),flush=True)
print("OPTIMIZED",json.dumps(result["optimized"]),flush=True)
print("SAVED",p,flush=True)
