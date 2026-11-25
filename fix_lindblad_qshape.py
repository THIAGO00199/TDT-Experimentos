from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/tdt_quantum_lindblad_2q1mode.py")
s=p.read_text().replace('return np.einsum("ijkabk->ijab",t)','return np.einsum("ijkabk->ijab",t).reshape(4,4)')
p.write_text(s)
print("fixed reduced qubit matrix shape")
