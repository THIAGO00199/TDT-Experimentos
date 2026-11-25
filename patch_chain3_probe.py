p="/home/iknowyou/TDT_Experimentos/tdt_quantum_chain3_control_return.py"
s=open(p).read()
needle="c0=np.zeros(12)\nbasepsi,basew,basee=evolve(c0)"
s=s.replace(needle,needle+'\nprint("PROBE",objective(np.ones(12)*0.5),objective(np.array([0.5,-0.3,0.2]*4)),flush=True)')
open(p,"w").write(s)
print("probe patched")
