from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/tdt_quantum_lindblad_2q1mode.py")
s=p.read_text()
start=s.index("def partial_trace(r, keep):")
end=s.index("def entropy(r):",start)
new='''def partial_trace(r, keep):
    t=r.reshape(2,2,NOSC,2,2,NOSC)
    if keep==[0,1]:
        return np.einsum("ijkabl->ijab",t) if False else np.einsum("ijkabk->ijab",t)
    if keep==[2]:
        return np.einsum("ijkijc->kc",t)
    raise ValueError("unsupported subsystem")
'''
s=s[:start]+new+s[end:]
p.write_text(s)
print("fixed partial trace")
