#include <bits/stdc++.h>
using namespace std;
static uint64_t sm(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct RNG{uint64_t s;RNG(uint64_t v):s(v){}uint64_t next(){return sm(s);}int idx(int n){return int(next()%n);}};
int main(){
 const int baths[]={8,16,32,64}, times[]={0,1,2,5,10,20};
 const int NSAMP=100000,H=12,CTRL=50,NS=8;
 cout<<setprecision(10)<<"Creutz size scaling; Nsystem=8; bath=8,16,32,64; H=12; control sweeps=50; trajectories=100000\n";
 cout<<"Creutz demon explicitly included in conserved energy; ring bonds J=2 within system, J=1 elsewhere incl interfaces.\n";
 for(int nb:baths){int N=NS+nb; vector<int> J(N,1);for(int i=0;i<NS-1;i++)J[i]=2;
   RNG irng(20261113ULL+nb*313);vector<int8_t>x0(N);for(int i=0;i<N;i++)x0[i]=(irng.next()&1)?1:-1;
   long E0=0;for(int i=0;i<N;i++)E0-=J[i]*x0[i]*x0[(i+1)%N];
   cout<<"\nBATH="<<nb<<" N="<<N<<" E_spin0="<<E0<<" E_total0="<<E0<<" X0=";
   for(auto s:x0)cout<<(s>0?'1':'0');cout<<"\n";
   for(int mode=0;mode<2;mode++){
    cout<<"CONTROL="<<(mode==0?"SYSTEM_8":"ALL_SPINS")<<"\n";
    for(int ts:times){
      long succ=0;long double sw=0,sw2=0,ovsum=0,dsum=0;long long ar=0,ac=0;
      RNG rng(20261113ULL+nb*1000003ULL+ts*1009ULL+mode*900000001ULL);
      for(int tr=0;tr<NSAMP;tr++){
       vector<int8_t>s=x0;int demon=0;
       auto attempt=[&](int i,bool field,int nfield){
        int l=(i+N-1)%N,r=(i+1)%N;
        int dE=2*s[i]*(J[l]*s[l]+J[i]*s[r]);
        if(field&&i<nfield)dE+=2*H*x0[i]*s[i];
        if(dE<=demon){demon-=dE;s[i]=-s[i];return true;}return false;
       };
       for(int k=0;k<ts*N;k++)if(attempt(rng.idx(N),false,0))ar++;
       int nf=(mode==0?NS:N),ob=0;for(int i=0;i<nf;i++)ob+=x0[i]*s[i];
       for(int k=0;k<CTRL*N;k++)if(attempt(rng.idx(N),true,nf))ac++;
       int oa=0;for(int i=0;i<nf;i++)oa+=x0[i]*s[i];
       double w=H*(oa-ob);sw+=w;sw2+=(long double)w*w;dsum+=demon;
       int ov=0;for(int i=0;i<N;i++)if(s[i]==x0[i])ov++;
       ovsum+=ov;if(ov==N)succ++;
      }
      double p=double(succ)/NSAMP, mw=double(sw/NSAMP),var=max(0.0,double(sw2/NSAMP)-mw*mw);
      cout<<"t="<<ts<<" exact="<<succ<<"/"<<NSAMP<<" p="<<p
          <<" KL95lower_zero="<<(succ? -1.0:log(double(NSAMP)/3.0))
          <<" W="<<mw<<" SE="<<sqrt(var/NSAMP)<<" demon="<<double(dsum/NSAMP)
          <<" overlap="<<double(ovsum/NSAMP)<<" acc_relax="<<double(ar)/NSAMP
          <<" acc_ctrl="<<double(ac)/NSAMP<<"\n";
    }
   }
 }
}