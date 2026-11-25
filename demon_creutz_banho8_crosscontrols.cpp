#define main creutz_original_main
#include "demon_creutz_banho8.cpp"
#undef main

static vector<double> readArray(const string&json,const string&key){
    string tag="\""+key+"\":[";
    size_t p=json.find(tag);
    if(p==string::npos)throw runtime_error("missing json key "+key);
    p+=tag.size();size_t q=json.find(']',p);
    if(q==string::npos)throw runtime_error("malformed array "+key);
    string body=json.substr(p,q-p);
    vector<double> v;stringstream ss(body);string item;
    while(getline(ss,item,',')){if(!item.empty())v.push_back(stod(item));}
    return v;
}
struct Paired {
    double mean=0,se=0,meanPos=0,sePos=0,meanRet=0,seRet=0;
    array<double,4> deMean{},deSE{};
    array<double,NS*NB> xcMean{},xcSE{};
    array<double,NB-1> bathMean{},bathSE{};
    array<double,NS-1> sysMean{},sysSE{};
};
static pair<double,double> meanSE(const vector<double>&v){
    double m=accumulate(v.begin(),v.end(),0.0)/v.size(),s=0;
    for(double x:v)s+=(x-m)*(x-m);
    return {m,sqrt(s/(v.size()-1)/v.size())};
}
static Paired compare(const Dist&A,const Dist&B,const vector<double>&ctrl,uint64_t seed){
    vector<double> dw,dpos,dret;dw.reserve(NVALID);dpos.reserve(NVALID);dret.reserve(NVALID);
    array<double,4> es{},es2{};
    array<double,NS*NB> xs{},xs2{};
    array<double,NB-1> bs{},bs2{};
    array<double,NS-1> ss{},ss2{};
    for(int j=0;j<NVALID;j++){
        uint64_t z=seed+uint64_t(j)*GOLD;
        Traj a=trajectory(A,ctrl,z),b=trajectory(B,ctrl,z);
        dw.push_back(b.work-a.work);dpos.push_back(b.wpos-a.wpos);dret.push_back(b.wneg-a.wneg);
        array<double,4> de={
            (b.sysE1-b.sysE0)-(a.sysE1-a.sysE0),
            (b.bathE1-b.bathE0)-(a.bathE1-a.bathE0),
            (b.intE1-b.intE0)-(a.intE1-a.intE0),
            (b.d1-b.d0)-(a.d1-a.d0)};
        for(int i=0;i<4;i++){es[i]+=de[i];es2[i]+=de[i]*de[i];}
        for(int i=0;i<NS*NB;i++){double x=b.xc1[i]-a.xc1[i];xs[i]+=x;xs2[i]+=x*x;}
        for(int i=0;i<NB-1;i++){double x=b.bc1[i]-a.bc1[i];bs[i]+=x;bs2[i]+=x*x;}
        for(int i=0;i<NS-1;i++){double x=b.c1[i]-a.c1[i];ss[i]+=x;ss2[i]+=x*x;}
    }
    Paired p;
    auto w=meanSE(dw),wi=meanSE(dpos),wr=meanSE(dret);
    p.mean=w.first;p.se=w.second;p.meanPos=wi.first;p.sePos=wi.second;p.meanRet=wr.first;p.seRet=wr.second;
    for(int i=0;i<4;i++){
        p.deMean[i]=es[i]/NVALID;
        double v=max(0.0,(es2[i]-NVALID*p.deMean[i]*p.deMean[i])/(NVALID-1));
        p.deSE[i]=sqrt(v/NVALID);
    }
    for(int i=0;i<NS*NB;i++){
        p.xcMean[i]=xs[i]/NVALID;
        double v=max(0.0,(xs2[i]-NVALID*p.xcMean[i]*p.xcMean[i])/(NVALID-1));
        p.xcSE[i]=sqrt(v/NVALID);
    }
    for(int i=0;i<NB-1;i++){
       p.bathMean[i]=bs[i]/NVALID;double v=max(0.0,(bs2[i]-NVALID*p.bathMean[i]*p.bathMean[i])/(NVALID-1));p.bathSE[i]=sqrt(v/NVALID);
    }
    for(int i=0;i<NS-1;i++){
       p.sysMean[i]=ss[i]/NVALID;double v=max(0.0,(ss2[i]-NVALID*p.sysMean[i]*p.sysMean[i])/(NVALID-1));p.sysSE[i]=sqrt(v/NVALID);
    }
    return p;
}
static void printPair(const string&name,const Paired&p){
    cout<<setprecision(12)<<name<<" dW="<<p.mean<<" SE="<<p.se
        <<" dWpos="<<p.meanPos<<" SEpos="<<p.sePos
        <<" dWreturn="<<p.meanRet<<" SEreturn="<<p.seRet
        <<" dEsys="<<p.deMean[0]<<" dEbath="<<p.deMean[1]
        <<" dEinterface="<<p.deMean[2]<<" dEdemon="<<p.deMean[3]<<"\n";
    double maxz=0;int maxi=0;double norm=0;
    for(int i=0;i<NS*NB;i++){double z=abs(p.xcMean[i])/max(p.xcSE[i],1e-15);if(z>maxz){maxz=z;maxi=i;}norm+=p.xcMean[i]*p.xcMean[i];}
    cout<<name<<" crosscorr_norm="<<sqrt(norm)<<" max_abs_z="<<maxz
        <<" max_pair=("<<maxi/NB<<","<<maxi%NB<<") delta="<<p.xcMean[maxi]
        <<" se="<<p.xcSE[maxi]<<"\n";
}
int main(){
    Dist A=makeA(),B=makeB(A);
    ifstream fi("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_resultado.json");
    string json((istreambuf_iterator<char>(fi)),istreambuf_iterator<char>());
    vector<double> ca=readArray(json,"control_A_coefficients"),cb=readArray(json,"control_B_coefficients");
    if(ca.size()!=NS*K||cb.size()!=NS*K)throw runtime_error("bad control coefficient length");
    Paired usingA=compare(A,B,ca,0x9e3779b97f4a7c15ULL);
    Paired usingB=compare(A,B,cb,0x243f6a8885a308d3ULL);
    printPair("SAME_A_CONTROL",usingA);printPair("SAME_B_CONTROL",usingB);
    ofstream f("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_crosscontrols.json");
    f<<setprecision(17)<<"{\n\"validation_trajectories\":"<<NVALID<<",\n";
    auto emit=[&](const string&name,const Paired&p){
      f<<"\""<<name<<"\":{\"dW_B_minus_A\":"<<p.mean<<",\"se_dW\":"<<p.se
       <<",\"dW_input_B_minus_A\":"<<p.meanPos<<",\"se_input\":"<<p.sePos
       <<",\"dW_return_B_minus_A\":"<<p.meanRet<<",\"se_return\":"<<p.seRet
       <<",\"delta_energy_components\":[";
      for(int i=0;i<4;i++)f<<"{\"mean\":"<<p.deMean[i]<<",\"se\":"<<p.deSE[i]<<"}"<<(i==3?"":",");
      f<<"],\"system_bath_cross_corr\":[";
      for(int i=0;i<NS*NB;i++)f<<"{\"mean\":"<<p.xcMean[i]<<",\"se\":"<<p.xcSE[i]<<"}"<<(i==NS*NB-1?"":",");
      f<<"],\"bath_bond_corr_deltas\":[";
      for(int i=0;i<NB-1;i++)f<<"{\"mean\":"<<p.bathMean[i]<<",\"se\":"<<p.bathSE[i]<<"}"<<(i==NB-2?"":",");
      f<<"],\"system_bond_corr_deltas\":[";
      for(int i=0;i<NS-1;i++)f<<"{\"mean\":"<<p.sysMean[i]<<",\"se\":"<<p.sysSE[i]<<"}"<<(i==NS-2?"":",");
      f<<"]}";
    };
    emit("A_control_applied_to_A_and_B",usingA);f<<",\n";emit("B_control_applied_to_A_and_B",usingB);f<<"\n}\n";
    cout<<"SAVED /home/iknowyou/TDT_Experimentos/demon_creutz_banho8_crosscontrols.json\n";
}
