import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply
import json, time
from pathlib import Path

Nosc=3
dims=[2,2,Nosc,Nosc,Nosc]
D=int(np.prod(dims))
I2=identity(2,format="csr",dtype=complex)
Io=identity(Nosc,format="csr",dtype=complex)
def embed(op, site):
    out=None
    for k,d in enumerate(dims):
        x=op if k==site else identity(d,format="csr",dtype=complex)
        out=x if out is None else kron(out,x,format="csr")
    return out
sz=csr_matrix(np.array([[1,0],[0,-1]],complex))
sm=csr_matrix(np.array([[0,1],[0,0]],complex))
sp=sm.getH()
a=csr_matrix(np.diag(np.sqrt(np.arange(1,Nosc)),1).astype(complex))
adag=a.getH()
H=csr_matrix((D,D),dtype=complex)
wq=[1.0,1.13]
for i in range(2): H += 0.5*wq[i]*embed(sz,i)
H += 0.04*(embed(sz,0)@embed(sz,1))
wa=[1.0,1.04,0.97]
for j in range(3): H += wa[j]*embed(adag@a,j+2)
Jchain=0.22
for j in range(2):
    H += Jchain*(embed(adag,j+2)@embed(a,j+3)+embed(a,j+2)@embed(adag,j+3))
g=0.22
H += g*(embed(sp,1)@embed(a,2)+embed(sm,1)@embed(adag,2))
H=H.tocsr()

def full_state(q):
    bath=np.zeros(Nosc**3,complex); bath[0]=1
    return np.kron(np.asarray(q,complex),bath)
bell=np.array([1,0,0,1],complex)/np.sqrt(2)
psi_bell=full_state(bell)
psi0=full_state(np.array([1,0,0,0],complex))
psi1=full_state(np.array([0,1,0,0],complex))
T=50.0; steps=501
A=(-1j)*H
t0=time.time()
traj_b=expm_multiply(A,psi_bell,start=0,stop=T,num=steps,endpoint=True)
traj_0=expm_multiply(A,psi0,start=0,stop=T,num=steps,endpoint=True)
traj_1=expm_multiply(A,psi1,start=0,stop=T,num=steps,endpoint=True)

def rho_red(psi, leftdim):
    mat=psi.reshape(leftdim,-1)
    return mat@mat.conj().T
def entropy(rho):
    ev=np.linalg.eigvalsh((rho+rho.conj().T)/2)
    ev=np.clip(ev,0,1)
    ev=ev[ev>1e-14]
    return float(-np.sum(ev*np.log(ev)))
def concurrence(rho):
    sy=np.array([[0,-1j],[1j,0]],complex)
    YY=np.kron(sy,sy)
    R=rho@YY@rho.conj()@YY
    vals=np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(R)),0,None)))[::-1]
    return float(max(0.0,vals[0]-sum(vals[1:])))
times=np.linspace(0,T,steps)
td=[]; conc=[]; sq=[]; sb=[]; mi=[]
for x,y,z in zip(traj_0,traj_1,traj_b):
    r0=rho_red(x,4); r1=rho_red(y,4); rb=rho_red(z,4)
    td.append(float(0.5*np.sum(np.abs(np.linalg.eigvalsh(r0-r1)))))
    conc.append(concurrence(rb)); sq.append(entropy(rb))
    bath=rho_red(z.reshape(4,-1).T.reshape(-1),Nosc**3) if False else None
    # For pure global state, S(bath)=S(qubits), and I(Q:B)=2*S(Q).
    mi.append(2*entropy(rb))
td=np.array(td); dtd=np.diff(td)/np.diff(times)
back=np.maximum(dtd,0)
blp=float(np.sum(back*np.diff(times)))
peaks=[]
for i in range(1,len(td)-1):
    if td[i]>=td[i-1] and td[i]>td[i+1] and td[i]>0.02:
        peaks.append({"t":float(times[i]),"trace_distance":float(td[i])})
out={
 "model":"2 qubits + 3 explicit harmonic modes; closed unitary chain; no Lindblad reservoir",
 "parameters":{"oscillator_cutoff":Nosc,"hilbert_dimension":D,"wq":wq,"wa":wa,"g":g,"J_chain":Jchain,"Jzz":0.04,"T":T,"steps":steps},
 "initial_pair":"|00> vs |01>, bath vacuum; trace distance of reduced two-qubit states",
 "initial_bell":"(|00>+|11>)/sqrt(2) tensor bath vacuum",
 "blp_positive_variation":blp,
 "trace_distance_initial":float(td[0]),"trace_distance_final":float(td[-1]),
 "trace_distance_max":float(td.max()),"backflow_peaks":peaks[:20],
 "bell_concurrence_initial":float(conc[0]),"bell_concurrence_min":float(np.min(conc)),"bell_concurrence_final":float(conc[-1]),
 "bell_qubit_entropy_final_nats":float(sq[-1]),"bell_mutual_information_final_nats":float(mi[-1]),
 "samples":[{"t":float(times[i]),"trace_distance":float(td[i]),"concurrence":float(conc[i]),"S_qubits":float(sq[i]),"I_QB":float(mi[i])} for i in range(0,steps,10)],
 "runtime_s":time.time()-t0
}
base=str(Path(__file__).resolve().parent / "tdt_quantum_chain3_backflow")
with open(base+".json","w") as f: json.dump(out,f,indent=2)
with open(base+".txt","w") as f:
    f.write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
print("SAVED",base+".json",base+".txt")
