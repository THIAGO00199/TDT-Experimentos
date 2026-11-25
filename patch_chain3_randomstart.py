p="/home/iknowyou/TDT_Experimentos/tdt_quantum_chain3_control_return.py"
s=open(p).read()
s=s.replace("x=np.zeros(12); best=x.copy(); bestf=objective(x)","x=rng.uniform(-0.25,0.25,size=12); best=x.copy(); bestf=objective(x)")
open(p,"w").write(s)
print("random-start patched")
