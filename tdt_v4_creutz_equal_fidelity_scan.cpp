#include <bits/stdc++.h>
using namespace std;
static uint64_t sm(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct RNG{uint64_t s;RNG(uint64_t v):s(v){}uint64_t next(){return sm(s);}int idx(int n){return int(next()%n);}};
struct Res{long long ss=0,ff=0;long double ws=0,wb=0,ws2=0,wb2=0,dem=0,over=0;};
int main(){
 const int baths[]={8,16,32,64}, amps[]={0,2,4,6,8,10,12,14,16,20};
 const int NSAMP=10000, HREL=20, CTRL=50, NS=8;
 cout<<setprecision(10)<<"Creutz amplitude scan; system fidelity target (first 8 spins); H scalar; t_relax=20; CTRL=50; coarse N=10000 per point\n";
 for(int nb:baths){int N=NS+nb;vector<int>J(N,1);for(int i=0;i<NS-1;i++)J[i]=2;
  vector<int8_t>x0(N);string sys="10101010";for(int i=0;i<NS;i++)x0[i]=(sys[i]=='1'?1:-1);
  RNG brng(9918273);for(int i=NS;i<N;i++)x0[i]=(brng.next()&1)?1:-1;
  cout<<"\nBATH "<<nb<<" N "<<N<<" X0 ";for(auto s:x0)cout<<(s>0?'1':'0');cout<<"\n";
  for(int mode=0;mode<2;mode++){
   cout<<"MODE "<<(mode==0?"SYSTEM_ONLY":"SYSTEM_AND_BATH")<<"\n";
   for(int H:amps){Res r;RNG rng(20261113ULL+nb*1000003ULL+H*101ULL+mode*900000001ULL);
    for(int tr=0;tr<NSAMP;tr++){
     vector<int8_t>s=x0;int demon=0;
     auto attempt=[&](int i,bool field){int l=(i+N-1)%N,rr=(i+1)%N;int dE=2*s[i]*(J[l]*s[l]+J[i]*s[rr]);if(field && (mode==1 || i<NS))dE+=2*H*x0[i]*s[i];if(dE<=demon){demon-=dE;s[i]=-s[i];return true;}return false;};
     for(int k=0;k<HREL*N;k++)attempt(rng.idx(N),false);
     int obS=0,obB=0;for(int i=0;i<NS;i++)obS+=x0[i]*s[i];for(int i=NS;i<N;i++)obB+=x0[i]*s[i];
     for(int k=0;k<CTRL*N;k++)attempt(rng.idx(N),true);
     int oaS=0,oaB=0;for(int i=0;i<NS;i++)oaS+=x0[i]*s[i];for(int i=NS;i<N;i++)oaB+=x0[i]*s[i];
     double wS=H*(oaS-obS),wB=(mode==1?H*(oaB-obB):0.0);
     r.ws+=wS;r.wb+=wB;r.ws2+=wS*wS;r.wb2+=wB*wB;r.dem+=demon;
     int ovS=0,ovF=0;for(int i=0;i<NS;i++)if(s[i]==x0[i])ovS++;for(int i=0;i<N;i++)if(s[i]==x0[i])ovF++;
     r.over+=ovS;if(ovS==NS)r.ss++;if(ovF==N)r.ff++;
    }
    double ms=double(r.ws/NSAMP),mb=double(r.wb/NSAMP);
    double ses=sqrt(max(0.0,double(r.ws2/NSAMP)-ms*ms)/NSAMP);
    double seb=sqrt(max(0.0,double(r.wb2/NSAMP)-mb*mb)/NSAMP);
    double p=double(r.ss)/NSAMP;
    cout<<"H="<<H<<" psys="<<p<<" sys_hits="<<r.ss<<" pfull="<<double(r.ff)/NSAMP
        <<" Wsys="<<ms<<" SEsys="<<ses<<" Wbath="<<mb<<" SEbath="<<seb
        <<" demon="<<double(r.dem/NSAMP)<<" overlapS="<<double(r.over/NSAMP)<<"\n";
   }
  }
 }
}