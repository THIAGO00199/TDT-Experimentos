from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_crosscontrols.cpp")
s=p.read_text()
s=s.replace("array<double,NS*NB> xcMean{},xcSE{};", "array<double,NS*NB> xcMean{},xcSE{};\n    array<double,NB-1> bathMean{},bathSE{};\n    array<double,NS-1> sysMean{},sysSE{};")
s=s.replace("array<double,NS*NB> xs{},xs2{};", "array<double,NS*NB> xs{},xs2{};\n    array<double,NB-1> bs{},bs2{};\n    array<double,NS-1> ss{},ss2{};")
s=s.replace("for(int i=0;i<NS*NB;i++){double x=b.xc1[i]-a.xc1[i];xs[i]+=x;xs2[i]+=x*x;}",
"""for(int i=0;i<NS*NB;i++){double x=b.xc1[i]-a.xc1[i];xs[i]+=x;xs2[i]+=x*x;}
        for(int i=0;i<NB-1;i++){double x=b.bc1[i]-a.bc1[i];bs[i]+=x;bs2[i]+=x*x;}
        for(int i=0;i<NS-1;i++){double x=b.c1[i]-a.c1[i];ss[i]+=x;ss2[i]+=x*x;}""")
s=s.replace("return p;\n}", """for(int i=0;i<NB-1;i++){
       p.bathMean[i]=bs[i]/NVALID;double v=max(0.0,(bs2[i]-NVALID*p.bathMean[i]*p.bathMean[i])/(NVALID-1));p.bathSE[i]=sqrt(v/NVALID);
    }
    for(int i=0;i<NS-1;i++){
       p.sysMean[i]=ss[i]/NVALID;double v=max(0.0,(ss2[i]-NVALID*p.sysMean[i]*p.sysMean[i])/(NVALID-1));p.sysSE[i]=sqrt(v/NVALID);
    }
    return p;
}""",1)
s=s.replace('<<" se="<<p.xcSE[maxi]<<"\\\\n";', 
'''<<" se="<<p.xcSE[maxi]<<"\\\\n";
    cout<<name<<" bath_delta=";
    for(int i=0;i<NB-1;i++)cout<<p.bathMean[i]<<"+/-"<<p.bathSE[i]<<(i==NB-2?"":";");
    cout<<" sys_delta=";
    for(int i=0;i<NS-1;i++)cout<<p.sysMean[i]<<"+/-"<<p.sysSE[i]<<(i==NS-2?"":";");
    cout<<"\\\\n";''')
s=s.replace('f<<"],\\\\"system_bath_cross_corr\\\\":[";', 'f<<"],\\\\"system_bath_cross_corr\\\\":[";')
# add arrays in JSON after cross-correlation list close
s=s.replace('f<<"]}";', '''f<<"],\\\\"bath_bond_corr_deltas\\\\":[";
      for(int i=0;i<NB-1;i++)f<<"{\\\\"mean\\\\":"<<p.bathMean[i]<<",\\\\"se\\\\":"<<p.bathSE[i]<<"}"<<(i==NB-2?"":",");
      f<<"],\\\\"system_bond_corr_deltas\\\\":[";
      for(int i=0;i<NS-1;i++)f<<"{\\\\"mean\\\\":"<<p.sysMean[i]<<",\\\\"se\\\\":"<<p.sysSE[i]<<"}"<<(i==NS-2?"":",");
      f<<"]}";''')
p.write_text(s)
print("patched cross-controls with paired bath/system bond-correlation differences")
