import numpy as np, json
from scipy.linalg import expm, svdvals
nu=.05; tau=.25; T=.8
rows=[]
for k in range(1,10):
    A=np.array([[0,1j*k],[nu*1j*k/tau,-1/tau]],dtype=complex)
    B=np.array([[1.],[0.]],dtype=complex)
    C=np.column_stack([B[:,0], A@B[:,0]])
    rank=np.linalg.matrix_rank(C,tol=1e-12)
    # Simpson quadrature for finite-horizon controllability Gramian
    n=4000; ts=np.linspace(0,T,n+1); h=T/n
    G=np.zeros((2,2),dtype=complex)
    for j,t in enumerate(ts):
        v=expm(A*t)@B
        weight=1 if j in (0,n) else (4 if j%2 else 2)
        G += weight*(v@v.conj().T)
    G*=h/3
    eig=np.linalg.eigvalsh(G).real
    rows.append(dict(k=k,rank=int(rank),gramian_eigenvalues=eig.tolist(),
        gramian_condition=float(eig[-1]/eig[0]),sigma_min_controllability=float(svdvals(C)[-1])))
print(json.dumps(dict(nu=nu,tau=tau,T=T,mode_results=rows),indent=2))
