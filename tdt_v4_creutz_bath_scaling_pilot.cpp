#include <bits/stdc++.h>
using namespace std;
static uint64_t sm(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct RNG{uint64_t s; RNG(uint64_t x):s(x){} uint64_t next(){return sm(s);} int idx(int n){return int(next()%n);}};
int main(){
 const int NS=8, NSAMP=100000, H=12, CTRL=10, times[]={0,1,5};
 const int baths[]={8,16,32,64};
 cout<<"CREUTZ BATH-SIZE SCALING PILOT\nNtraj="<<NSAMP<<" H="<<H<<" control_sweeps="<<CTRL<<"\n";
 cout<<"Model: ring; J_SS=2 on 7 internal system bonds, J_interface=1, J_BB=1; demon explicit; E_d0=0.\n";
 cout<<"WARNING: Wrec is measured; Wdiss=Wrec-DeltaF is NOT reported because isolated microcanonical model has no externally fixed T or canonical DeltaF.\n";
 for(int nb:baths){
  int N=NS+nb; vector<int> J(N,1); for(int i=0;i<NS-1;i++)J[i]=2;
  vector<int> x0(N); RNG ir(20261113); for(int i=0;i<N;i++)x0[i]=(ir.next()&1)?1:-1;
  long E0=0; for(int i=0;i<N;i++)E0-=J[i]*x0[i]*x0[(i+1)%N];
  cout<<"\nBATH="<<nb<<" TOTAL_SPINS="<<N<<" E_spin0="<<E0<<" X0=";
  for(int z:x0)cout<<(z>0?'1':'0'); cout<<"\n";
  for(int ts:times){
   long success=0; long double sw=0,sw2=0,dem=0,ov=0; long long ar=0,ac=0;
   RNG rng(0x9e3779b97f4a7c15ULL ^ (uint64_t)nb*1000003ULL ^ (uint64_t)ts*9176ULL);
   for(int tr=0;tr<NSAMP;tr++){
    vector<int8_t>s(N);for(int i=0;i<N;i++)s[i]=x0[i]; int demon=0;
    auto attempt=[&](int i,bool field){
      int l=(i+N-1)%N,r=(i+1)%N;
      int dE=2*s[i]*(J[l]*s[l]+J[i]*s[r]);
      if(field)dE+=2*H*x0[i]*s[i];
      if(dE<=demon){demon-=dE;s[i]=-s[i];return true;}return false;
    };
    for(int k=0;k<ts*N;k++)if(attempt(rng.idx(N),false))ar++;
    int before=0;for(int i=0;i<N;i++)before+=x0[i]*s[i];
    for(int k=0;k<CTRL*N;k++)if(attempt(rng.idx(N),true))ac++;
    int after=0,match=0;for(int i=0;i<N;i++){after+=x0[i]*s[i];if(s[i]==x0[i])match++;}
    double w=H*(after-before);sw+=w;sw2+=(long double)w*w;dem+=demon;ov+=match;
    if(match==N)success++;
   }
   double mw=(double)(sw/NSAMP), var=max(0.0,(double)(sw2/NSAMP)-mw*mw);
   cout<<"t="<<ts<<" exact="<<success<<"/"<<NSAMP<<" p="<<setprecision(8)<<(double)success/NSAMP
       <<" Wrec="<<mw<<" SE="<<sqrt(var/NSAMP)<<" demon_mean="<<(double)(dem/NSAMP)
       <<" mean_matching_spins="<<(double)(ov/NSAMP)<<"/"<<N
       <<" relax_accepts/traj="<<(double)ar/NSAMP<<" ctrl_accepts/traj="<<(double)ac/NSAMP<<"\n";
  }
 }
}