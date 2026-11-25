from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_crosscontrols.cpp")
s=p.read_text()
s=s.replace(r'\\"',r'\"')
p.write_text(s)
print("fixed C++ JSON quote escapes")
