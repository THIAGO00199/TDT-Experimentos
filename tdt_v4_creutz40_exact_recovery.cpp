#include <bits/stdc++.h>
using namespace std;
static uint64_t splitmix64(uint64_t &x){ uint64_t z=(x+=0x9e3779b97f4a7c15ULL); z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
struct RNG{ uint64_t s; RNG(uint64_t x):s(x){} uint64_t next(){return splitmix64(s);} int idx(int n){return (int)(next()%n);} };
int main(){
 const int N=40, NS=8; const int J[40]={2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
 const int sweeps[]={0,1,2,5,10,20}; const int NSAMP=100000, CONTROL_SWEEPS=50, DEMON0=0, H=12;
 int x0[N]; RNG initrng(20261113); for(int i=0;i<N;i++) x0[i]=(initrng.next()&1)?1:-1;
 auto energy=[&](const array<int8_t,N>&s){long e=0; for(int i=0;i<N;i++){int j=(i+1)%N; e-=J[i]*s[i]*s[j];} return e;};
 long E0=0; for(int i=0;i<N;i++) E0-=J[i]*x0[i]*x0[(i+1)%N];
 cout<<"MODEL Creutz demon, 40-spin ring; system=0..7, bath=8..39\n";
 cout<<"J_SS=2, J_BB=1, J_SB=1; E_d0="<<DEMON0<<"; H="<<H<<"; control_sweeps="<<CONTROL_SWEEPS<<"\n";
 cout<<"X0="; for(int i=0;i<N;i++) cout<<(x0[i]>0?'1':'0'); cout<<"; E_spin0="<<E0<<"; E_total="<<E0+DEMON0<<"\n";
 cout<<"Ntraj="<<NSAMP<<"; sweep=40 attempted single-spin updates\n";
 for(int mode=0;mode<2;mode++){
   cout<<"\nCONTROL "<<(mode==0?"SYSTEM_ONLY_8":"ALL_40")<<"\n";
   for(int ts:sweeps){
     long success=0; long double worksum=0, work2=0; long double demonSum=0, overlapSum=0; long acceptedRelax=0, acceptedCtrl=0;
     RNG rng(20261113ULL + (uint64_t)ts*1009 + mode*9000001ULL);
     for(int tr=0;tr<NSAMP;tr++){
       array<int8_t,N>s; for(int i=0;i<N;i++) s[i]=x0[i]; int demon=DEMON0;
       auto attempt=[&](int i, bool field, int ncontrol){
         int l=(i+N-1)%N, r=(i+1)%N;
         int dE=2*s[i]*(J[l]*s[l]+J[i]*s[r]);
         if(field && i<ncontrol) dE += 2*H*x0[i]*s[i]; // field energy -H*x0*s
         if(dE<=demon){ demon-=dE; s[i]=-s[i]; return true;} return false;
       };
       for(int k=0;k<ts*N;k++) if(attempt(rng.idx(N),false,0)) acceptedRelax++;
       int ncontrol=(mode==0?NS:N);
       int overlapBefore=0; for(int i=0;i<ncontrol;i++) overlapBefore+=x0[i]*s[i];
       // Instantaneously switch on target-aligned field, evolve, then switch it off.
       // Work_on=-H*overlapBefore; Work_off=+H*overlapAfter.
       int overlapAfter=0;
       for(int k=0;k<CONTROL_SWEEPS*N;k++) if(attempt(rng.idx(N),true,ncontrol)) acceptedCtrl++;
       for(int i=0;i<ncontrol;i++) overlapAfter+=x0[i]*s[i];
       double w=H*(overlapAfter-overlapBefore);
       worksum+=w; work2+=(long double)w*w; demonSum+=demon;
       int ov=0; for(int i=0;i<N;i++) if(s[i]==x0[i]) ov++;
       overlapSum+=ov;
       if(ov==N) success++;
     }
     double p=(double)success/NSAMP;
     double meanW=(double)(worksum/NSAMP);
     double varW=max(0.0,(double)(work2/NSAMP)-meanW*meanW);
     double seW=sqrt(varW/NSAMP);
     double meanD=(double)(demonSum/NSAMP);
     double meanOv=(double)(overlapSum/NSAMP);
     double klLower=(success==0?log((double)NSAMP/3.0):-log(p)); // 95% zero-count upper bound p<~3/N
     cout<<"t_sweeps="<<ts<<" success="<<success<<"/"<<NSAMP<<" p_exact="<<setprecision(8)<<p
         <<" KL_delta_to_final="<<(success? -log(p): -1.0)<<" KL95_lower_if_zero="<<(success? -1.0:klLower)
         <<" Wrec_signed="<<meanW<<" SE="<<seW<<" demon_mean="<<meanD<<" mean_overlap40="<<meanOv
         <<" accept_relax_per_traj="<<(double)acceptedRelax/NSAMP
         <<" accept_ctrl_per_traj="<<(double)acceptedCtrl/NSAMP<<"\n";
   }
 }
}
