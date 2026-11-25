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
start=time.time()
records=[]
for k in range(NSTEPS+1):
    if k%10==0 or k==NSTEPS:
        m=metrics(rho_vec); m["t"]=k*DT; m["E_total"]=energy(rho_vec,H0)
        records.append(m)
    if k<NSTEPS:
        rho_vec=propagate(rho_vec,H0,DT)
        # Suppress only floating-point anti-Hermitian/trace drift for recorded state;
        # no positivity clipping is performed.
        if (k+1)%20==0:
            rr=rho_vec.reshape((D,D),order="F")
            rr=(rr+rr.conj().T)/2
            rr/=np.trace(rr)
            rho_vec=rr.reshape(-1,order="F")
final=metrics(rho_vec)
result={
 "model":"2 qubits + one truncated harmonic oscillator mode + thermal Lindblad reservoir",
 "hilbert_dimensions":{"qubits":4,"oscillator":NOSC,"total":D,"density_matrix":D*D},
 "parameters":{"hbar":HBAR,"omega_q1":WQ1,"omega_q2":WQ2,"omega_osc":WA,"g_JC":G,
               "Jzz":JZZ,"kappa":KAPPA,"nbar":NBAR,"dt":DT,"steps":NSTEPS,"T":T},
 "initial_state":"Bell pair (|00>+|11>)/sqrt(2) tensor thermal oscillator state",
 "dissipation":"kappa(nbar+1)D[a] + kappa*nbar*D[a^dagger]",
 "control":"none in this baseline; pulse optimization is the next stage",
 "initial_metrics":records[0],"final_metrics":final,
 "time_series":records,
 "runtime_seconds":time.time()-start,
 "interpretation_warning":"S_qubits is not entanglement entropy when the qubit subsystem is mixed; report mutual information, negativity and concurrence separately."
}
path=OUT/"tdt_quantum_lindblad_2q1mode_strong.json"
path.write_text(json.dumps(result,indent=2))
txt=OUT/"tdt_quantum_lindblad_2q1mode_strong.txt"
txt.write_text(
"QUANTUM TDT — LINDBLAD BASELINE\n"
"2 qubits + one truncated harmonic oscillator mode + thermal Markov reservoir\n\n"
+json.dumps(result,indent=2)+"\n")
print("MODEL",result["model"])
print("DIMENSIONS",result["hilbert_dimensions"])
print("PARAMETERS",result["parameters"])
print("INITIAL",json.dumps(result["initial_metrics"]))
print("FINAL",json.dumps(result["final_metrics"]))
print("RUNTIME_SECONDS",result["runtime_seconds"])
print("SAVED",path)
