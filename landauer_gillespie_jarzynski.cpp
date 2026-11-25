#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
using namespace std;
struct Stats { double meanW,seW,jar,seJar,ess,finalP1,sdW,minW,maxW; };
Stats run(double tau, int N, uint64_t seed, double B) {
    const double beta=0.4, nu=1.0, epsf=15.0;
    const double k01=nu*exp(-beta*B), c=beta*epsf/tau, edot=epsf/tau;
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0,1.0);
    long double sw=0, sw2=0, sj=0, sj2=0, essnum=0;
    long long n1=0;
    double mn=1e100,mx=-1e100;
    for(int j=0;j<N;j++){
        int state=(U(rng)<0.5)?0:1;
        double t=0,w=0;
        while(true){
            double u=max(U(rng),1e-300), z=-log(u), dt;
            if(state==0) dt=z/k01;
            else {
                double rate=nu*exp(-beta*(B-edot*t));
                dt=log1p(c*z/rate)/c;
            }
            if(t+dt>=tau){
                if(state==1) w+=edot*(tau-t);
                break;
            }
            if(state==1) w+=edot*dt;
            t+=dt; state=1-state;
        }
        double weight=exp(-beta*w);
        sw+=w; sw2+=(long double)w*w;
        sj+=weight; sj2+=(long double)weight*weight;
        if(state==1)n1++;
        mn=min(mn,w);mx=max(mx,w);
    }
    double mW=(double)(sw/N), mJ=(double)(sj/N);
    double varW=(double)((sw2/N)-(long double)mW*mW);
    double varJ=(double)((sj2/N)-(long double)mJ*mJ);
    double seW=sqrt(max(0.0,varW)/(N-1));
    double seJ=sqrt(max(0.0,varJ)/(N-1));
    double ess=(double)(sj*sj/sj2);
    double meanFinal=(double)n1/N;
    return {mW,seW,mJ,seJ,ess,meanFinal,sqrt(max(0.0,varW)),mn,mx};
}
int main(){
    const double beta=0.4, kT=2.5, B=20.0, epsf=15.0;
    const int N=200000;
    const double tauRelax=1.0/(2.0*exp(-beta*B));
    const double dF=kT*log(2.0)-kT*log1p(exp(-beta*epsf));
    const double target=exp(-beta*dF);
    ofstream jf("landauer_gillespie_jarzynski.json");
    jf<<setprecision(12)<<"{\n  \"parameters\": {\"beta\":0.4,\"kT\":2.5,\"nu\":1,\"B\":20,\"epsilon_final\":15,\"N_per_protocol\":"<<N<<",\"tau_relax_initial\":"<<tauRelax<<"},\n  \"results\":[\n";
    vector<pair<string,double>> cases={{"FAST_eta_0.01",0.01*tauRelax},{"SLOW_eta_1000",1000.0*tauRelax}};
    for(int i=0;i<2;i++){
        auto t0=chrono::steady_clock::now();
        auto s=run(cases[i].second,N,20261113+i,B);
        double z=(s.jar-target)/s.seJar;
        double rel=100*(s.jar-target)/target;
        double runtime=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
        cout<<setprecision(12)<<cases[i].first<<" N="<<N<<" tau="<<cases[i].second
            <<" meanW="<<s.meanW<<" SEW="<<s.seW<<" SDW="<<s.sdW
            <<" Jarzynski="<<s.jar<<" SE="<<s.seJar<<" target="<<target
            <<" z_naive="<<z<<" relerr_pct="<<rel<<" ESS="<<s.ess
            <<" ESS_frac="<<s.ess/N<<" final_p1="<<s.finalP1
            <<" W_range=["<<s.minW<<","<<s.maxW<<"] runtime_s="<<runtime<<endl;
        jf<<"    {\"protocol\":\""<<cases[i].first<<"\",\"N\":"<<N<<",\"tau\":"<<cases[i].second
          <<",\"mean_work\":"<<s.meanW<<",\"mean_work_SE\":"<<s.seW<<",\"work_std\":"<<s.sdW
          <<",\"jarzynski_estimate\":"<<s.jar<<",\"jarzynski_SE_naive\":"<<s.seJar
          <<",\"jarzynski_target\":"<<target<<",\"jarzynski_z_naive\":"<<z
          <<",\"relative_error_percent\":"<<rel<<",\"weight_ESS\":"<<s.ess
          <<",\"weight_ESS_fraction\":"<<s.ess/N<<",\"final_p1\":"<<s.finalP1
          <<",\"work_min\":"<<s.minW<<",\"work_max\":"<<s.maxW<<",\"runtime_s\":"<<runtime<<"}"
          <<(i==0?",\n":"\n");
    }
    jf<<"  ],\n  \"DeltaF\":"<<dF<<",\n  \"Jarzynski_target\":"<<target<<"\n}\n";
    jf.close();
    cout<<"Saved landauer_gillespie_jarzynski.json"<<endl;
}
