#include <bits/stdc++.h>
using namespace std;
uint64_t splitmix(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct RNG{uint64_t x;RNG(uint64_t s):x(s){} uint64_t next(){return splitmix(x);} int idx(int n){return next()%n;}};
int main(){
 const int baths[]={8,16,32,64}; const double amps[]={0,0.5,1,2,4,8,12};
 const int NS=8, NTR=20000, HREL=20, CTRL=50, K=5, EMAX=20;
 const double profile[K]={0,0.5,1,0.5,0};
 cout<<setprecision(9)<<"Bounded Creutz demon, five-knot parametric fields; Emax="<<EMAX<<"; Ntraj="<<NTR<<"\n";
 for(int nb:baths){
  int N=NS+nb; vector<int> J(N,1);for(int i=0;i<NS-1;i++)J[i]=2;
  vector<int8_t>x0(N);string sys="10101010";for(int i=0;i<NS;i++)x0[i]=sys[i]=='1'?1:-1;
  RNG br(9918273);for(int i=NS;i<N;i++)x0[i]=(br.next()&1)?1:-1;
  cout<<"\nBATH "<<nb<<" N "<<N<<" X0 ";for(auto s:x0)cout<<(s>0?'1':'0');cout<<"\n";
  for(int mode=0;mode<2;mode++){
   cout<<"MODE "<<(mode==0?"SYSTEM_ONLY":"SYSTEM_PLUS_BATH")<<"\n";
   for(double A:amps){
    long long hits=0, full=0;long double ws=0,wb=0,cs=0,cb=0,dem=0,demcap=0;
    RNG rng(20261113ULL+nb*1000003ULL+(int)(A*1000)*101ULL+mode*900000001ULL);
    for(int tr=0;tr<NTR;tr++){
     vector<int8_t>s=x0;int d=0;double oldh=0;long double wsys=0,wbath=0;
     auto hval=[&](int i,int k){if(mode==0 && i>=NS)return 0.0;return A*profile[k]*x0[i];};
     // Relaxation at zero field
     for(int z=0;z<HREL*N;z++){
      int i=rng.idx(N),l=(i+N-1)%N,r=(i+1)%N;
      int de=2*s[i]*(J[l]*s[l]+J[i]*s[r]);
      if(de<=d && d-de<=EMAX){d-=de;s[i]=-s[i];}
     }
     int os=0,ob=0;for(int i=0;i<NS;i++)os+=x0[i]*s[i];for(int i=NS;i<N;i++)ob+=x0[i]*s[i];
     // 5-knot linear interpolation over 50 sweeps; include work at each knot change
     double prevH=0;
     for(int step=0;step<CTRL;step++){
      double u=double(step)/(CTRL-1)*(K-1);int k=min(K-2,(int)floor(u));double f=u-k;
      double g=profile[k]*(1-f)+profile[k+1]*f;
      double h=A*g;
      // field work dW=-sum x0_i*s_i dh_i, separately
      int ovS=0,ovB=0;for(int i=0;i<NS;i++)ovS+=x0[i]*s[i];for(int i=NS;i<N;i++)ovB+=x0[i]*s[i];
      wsys-=ovS*(h-prevH);
      if(mode==1)wbath-=ovB*(h-prevH);
      prevH=h;
      // One attempted flip per spin per sweep
      for(int z=0;z<N;z++){
       int i=rng.idx(N),l=(i+N-1)%N,r=(i+1)%N;
       int de=2*s[i]*(J[l]*s[l]+J[i]*s[r]);
       double hi=(i<NS||mode==1)?h*x0[i]:0.0;
       de+=(int)llround(2*hi*s[i]);
       if(de<=d && d-de<=EMAX){d-=de;s[i]=-s[i];}
      }
      // knot field held; next step performs parameter work
     }
     // ramp field back to zero; external work
     int ovS=0,ovB=0;for(int i=0;i<NS;i++)ovS+=x0[i]*s[i];for(int i=NS;i<N;i++)ovB+=x0[i]*s[i];
     wsys-=ovS*(0-prevH);if(mode==1)wbath-=ovB*(0-prevH);
     int fs=0,ff=0;for(int i=0;i<NS;i++)if(s[i]==x0[i])fs++;for(int i=0;i<N;i++)if(s[i]==x0[i])ff++;
     if(fs==NS)hits++;if(ff==N)full++;
     ws+=wsys;wb+=wbath;cs+=A*A*0.5*CTRL;cb+=(mode==1?A*A*0.5*CTRL*nb:0);dem+=d;if(d>=EMAX)demcap++;
    }
    double p=double(hits)/NTR,ms=double(ws/NTR),mb=double(wb/NTR);
    double ses=sqrt(max(0.0,double(0)) ); // SE omitted for compact scan
    cout<<"A="<<A<<" p_sys="<<p<<" hits="<<hits<<" p_full="<<double(full)/NTR
        <<" Wsys="<<ms<<" Wbath="<<mb<<" Csys="<<double(cs/NTR)<<" Cbath="<<double(cb/NTR)
        <<" demon_mean="<<double(dem/NTR)<<" cap_fraction="<<double(demcap/NTR)<<"\n";
   }
  }
 }
}