import numpy as np, json, time
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.legendre import legval
from scipy.optimize import minimize

Nosc=3; dims=[2,2,Nosc,Nosc,Nosc]; D=int(np.prod(dims))
def embed(op,site):
    out=None
    for k,d in enumerate(dims):
        x=op if k==site else identity(d,format="csr",dtype=complex)
        out=x if out is None else kron(out,x,format="csr")
    return out
sz=csr_matrix(np.array([[1,0],[0,-1]],complex))
sm=csr_matrix(np.array([[0,1],[0,0]],complex)); sp=sm.getH()
sx=csr_matrix(np.array([[0,1],[1,0]],complex))
sy=csr_matrix(np.array([[0,-1j],[1j,0]],complex))
a=csr_matrix(np.diag(np.sqrt(np.arange(1,Nosc)),1).astype(complex)); adag=a.getH()
H=csr_matrix((D,D),dtype=complex)
for i,w in enumerate([1.0,1.13]): H+=0.5*w*embed(sz,i)
H+=0.04*(embed(sz,0)@embed(sz,1))
for j,w in enumerate([1.0,1.04,0.97]): H+=w*embed(adag@a,j+2)
for j in range(2): H+=0.22*(embed(adag,j+2)@embed(a,j+3)+embed(a,j+2)@embed(adag,j+3))
H+=0.22*(embed(sp,1)@embed(a,2)+embed(sm,1)@embed(adag,2))
H=H.tocsr()
ops=[embed(sx,0),embed(sy,0),embed(sx,1),embed(sy,1)]
bell=np.array([1,0,0,1],complex)/np.sqrt(2)
bath=np.zeros(Nosc**3,complex); bath[0]=1
psi0=np.kron(bell,bath)
def redq(psi):
    z=psi.reshape(4,-1); return z@z.conj().T
def fidelity_bell(psi):
    r=redq(psi); return float(np.real(np.vdot(bell,r@bell)))
def entropy(rho):
    ev=np.linalg.eigvalsh((rho+rho.conj().T)/2); ev=np.clip(ev,0,1); ev=ev[ev>1e-14]
    return float(-np.sum(ev*np.log(ev)))
# Free evolution to the start of the first measured return window.
dt=0.08; t_start=1.8; t_end=3.4
psi_start=expm_multiply((-1j)*H*t_start,psi0)
nsteps=int(round((t_end-t_start)/dt)); dt=(t_end-t_start)/nsteps
def controls(c,s):
    # c shape (4,3); smooth envelope vanishes at both endpoints
    x=2*s-1
    return np.sin(np.pi*s)*np.array([legval(x,c[j]) for j in range(4)])
def evolve(c):
    psi=psi_start.copy(); work=0.0; effort=0.0
    us=[]
    for k in range(nsteps):
        s=(k+0.5)/nsteps
        u=controls(c.reshape(4,3),s)
        Ht=H.copy()
        for j in range(4): Ht=Ht+u[j]*ops[j]
        # signed work by the driven fields: integral dot(u)<O> dt
        # midpoint expectation and finite-difference pulse derivative
        eps=1e-5
        up=controls(c.reshape(4,3),min(1,s+eps))
        um=controls(c.reshape(4,3),max(0,s-eps))
        dudt=(up-um)/(2*eps*(t_end-t_start))
        exps=np.array([np.real(np.vdot(psi,op@psi)) for op in ops])
        work += float(np.dot(dudt,exps))*dt
        effort += float(np.dot(u,u))*dt
        psi=expm_multiply((-1j)*Ht*dt,psi)
        us.append(u)
    return psi,work,effort
def objective(c):
    psi,w,e=evolve(c)
    return 1-fidelity_bell(psi)+0.002*e
c0=np.zeros(12)
basepsi,basew,basee=evolve(c0)
print("PROBE",objective(np.ones(12)*0.5),objective(np.array([0.5,-0.3,0.2]*4)),flush=True)
start=time.time()
rng=np.random.default_rng(20261113)
x=rng.uniform(-0.25,0.25,size=12); best=x.copy(); bestf=objective(x)
for k in range(1,101):
    ak=0.18/(k**0.602); ck=0.12/(k**0.101)
    delta=rng.choice([-1.0,1.0],size=12)
    fp=objective(np.clip(x+ck*delta,-1.5,1.5))
    fm=objective(np.clip(x-ck*delta,-1.5,1.5))
    grad=((fp-fm)/(2*ck))*delta
    x=np.clip(x-ak*grad,-1.5,1.5)
    fx=objective(x)
    if fx<bestf: bestf=fx; best=x.copy()
    if k%20==0: print("SPSA",k,"current",fx,"best",bestf,flush=True)
res=type("Result",(),{"x":best,"success":True,"message":"SPSA completed","nit":100,"fun":bestf})()
optpsi,optw,opte=evolve(res.x)
out={
 "model":"closed unitary two-qubit + 3-mode harmonic chain; no Lindblad bath",
 "control_window":[t_start,t_end],"dt":dt,"nsteps":nsteps,
 "backflow_window_basis":"first return of reduced two-qubit trace distance, local peak near t=2.8 in free evolution",
 "control":"four local X/Y drives, one smooth sin(pi*s) envelope times quadratic Legendre polynomial per channel; 12 coefficients",
 "optimizer":"SPSA 100 iterations, fixed seed, 12 coefficients clipped to [-1.5,1.5]",
 "objective":"1 - reduced-qubit Bell fidelity + 0.002 * integral(sum_j u_j(t)^2)dt",
 "baseline":{"bell_fidelity":fidelity_bell(basepsi),"infidelity":1-fidelity_bell(basepsi),"signed_control_work":basew,"effort_integral_u2":basee,"S_qubits":entropy(redq(basepsi))},
 "optimized":{"bell_fidelity":fidelity_bell(optpsi),"infidelity":1-fidelity_bell(optpsi),"signed_control_work":optw,"effort_integral_u2":opte,"success":bool(res.success),"message":str(res.message),"iterations":int(res.nit),"objective":float(res.fun),"coefficients":res.x.reshape(4,3).tolist(),"S_qubits":entropy(redq(optpsi))},
 "runtime_s":time.time()-start
}
base="/home/iknowyou/TDT_Experimentos/tdt_quantum_chain3_control_return"
with open(base+".json","w") as f: json.dump(out,f,indent=2)
with open(base+".txt","w") as f: f.write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2)); print("SAVED",base+".json",base+".txt")
