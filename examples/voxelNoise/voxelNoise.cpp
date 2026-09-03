// sigma(N): how much of the answer is Monte-Carlo noise, per arm.
//
// The same configuration is run with K different seeds and the SPREAD of the
// resulting surface-height field is measured. Both arms are seeded explicitly
// (the level-set arm defaults to useRandomSeeds = true, which would otherwise
// give it scatter the voxel arm does not have).
//
// The height field is binned on the lattice pitch so both arms report the
// same quantity: h(x) over the trench floor band.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>
#include <psDomain.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeTrench.hpp>
#include <process/psProcess.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <cstdlib>
namespace ls=viennals; namespace cs=viennacs; namespace ps=viennaps;
using T=double; constexpr int D=2;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

static T GD=1.0;                 // set per sweep point
static T W=40., MASKH=30.;   // settable: feature resolution W/delta matters
static const T BINW=1.0;         // FIXED physical bin: comparing
                                 // sigma_field across Delta needs
                                 // one statistic, not one per grid
static T BLO=-0.35*W/2, BHI=0.35*W/2;   // floor band
// A blanket (unmasked, flat) surface is the control: with no curvature every
// probe convention coincides exactly, so any residual arm-to-arm difference
// there is rate, not reading.
static bool BLANKET=false;
// converge coverages on the initial surface before stepping, as the LS arm does
static bool INITCOV=false;
static T COSPOW=1.0;   // neutral source cosine power (1 = diffuse)
static int INITSWEEPS=0;   // reported back for the last voxel run

// h(x) binned on the grid pitch, so both arms report the same field
using Field = std::map<int, T>;
static int binOf(T x){ return (int)std::floor(x/BINW); }

static ps::SmartPointer<ps::Domain<T,D>> makeDomain(){
  auto dom=ps::SmartPointer<ps::Domain<T,D>>::New(GD,T(4)*W,T(4)*W);
  if(BLANKET) ps::MakePlane<T,D>(dom,GD,T(4)*W,T(4)*W,T(0),false,
                                 ps::Material::Si).apply();
  else ps::MakeTrench<T,D>(dom,W,T(0),T(0),MASKH,T(0),false,
                           ps::Material::Si,ps::Material::Mask).apply();
  return dom;
}

// h(x) over the floor band, from the level-set surface mesh
static Field lsField(ps::SmartPointer<ps::Domain<T,D>> dom,size_t *nodes=nullptr){
  auto mesh=ps::SmartPointer<ls::Mesh<T>>::New();
  ls::ToSurfaceMesh<T,D>(dom->getLevelSets().back(),mesh).apply();
  if(nodes) *nodes=mesh->getNodes().size();
  std::map<int,std::pair<T,int>> acc;
  for(const auto&n:mesh->getNodes()){
    if(n[0]<BLO||n[0]>BHI) continue;
    auto &a=acc[binOf(n[0])]; a.first+=n[1]; ++a.second;
  }
  Field h;
  for(const auto&kv:acc) h[kv.first]=kv.second.first/kv.second.second;
  return h;
}

// The DEPTH etched, h(t) - h(0), not the absolute height: the two arms read
// the surface by different conventions (mesh nodes against a fill-weighted
// column top), so only the displacement is a shared quantity. The baseline is
// seed-independent, so subtracting it leaves every sigma below unchanged.
static Field lsArm(ps::ChemicalMechanism<T> mech,T time,unsigned rays,
                   unsigned seed,double &secs,size_t *nodes=nullptr){
  auto dom=makeDomain();
  auto model=ps::SmartPointer<ps::SurfaceChemistry<T,D>>::New(mech);
  ps::Process<T,D> proc(dom,model,time);
  proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
  ps::RayTracingParameters rt;
  rt.raysPerPoint=rays;
  rt.useRandomSeeds=false;      // the whole point: seed it explicitly
  rt.rngSeed=seed;
  proc.setParameters(rt);
  ps::CoverageParameters cov; cov.tolerance=1e-6; cov.maxIterations=40;
  proc.setParameters(cov);
  const Field before=lsField(dom,nodes);
  const auto t0=std::chrono::steady_clock::now();
  proc.apply();
  secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  Field h=lsField(dom);
  for(auto &kv:h){ auto it=before.find(kv.first); if(it!=before.end()) kv.second-=it->second; }
  return h;
}

static Field voxelArm(ps::ChemicalMechanism<T> mech,T time,int steps,
                      size_t rays,unsigned seed,double &secs,
                      size_t *cells=nullptr){
  auto dom=makeDomain();
  auto topLS=dom->getLevelSets().back();
  auto deep=ls::SmartPointer<ls::Domain<T,D>>::New(topLS->getGrid());
  { T o[D]={0.,-36.}, n[D]={0.,1.};
    ls::MakeGeometry<T,D>(deep,ls::SmartPointer<ls::Plane<T,D>>::New(o,n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T,D>>> lss{deep};
  auto mm=ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::Si);
  for(size_t l=0;l<dom->getLevelSets().size();++l){
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto cellSet=viennacore::SmartPointer<cs::DenseCellSet<T,D>>::New();
  cellSet->setCellSetPosition(true);
  cellSet->setCoverMaterial((int)ps::Material::GAS);
  cellSet->fromLevelSets(lss,mm,BLANKET?T(10):MASKH+4.);
  cs::LatticeMap<T,D> lat(*cellSet);
  const auto &mid=*cellSet->getScalarData("Material");
  std::vector<T> fill(cellSet->getNumberOfCells(),T(0));
  std::vector<int> material(cellSet->getNumberOfCells());
  for(size_t c=0;c<fill.size();++c){
    material[c]=(int)mid[c];
    fill[c]= material[c]==(int)ps::Material::GAS?T(0):T(1);
  }
  ps::VoxelChemistry<T,D> vox(mech,lat,fill,material);
  vox.setRaysPerCell(rays);   // per SURFACE CELL, as the LS arm is per point
  vox.setNeutralCosinePower(COSPOW);
  vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
  auto cov=vox.makeCoverages();
  if(INITCOV) INITSWEEPS=vox.initialiseCoverages(cov,seed,100,T(1e-6));
  if(cells) *cells=vox.surfaceCellCount();
  auto probe=[&](){
    const auto &dims=lat.dims();
    Field h;
    for(int i=0;i<dims[0];++i){
      const T x=lat.minCorner()[0]+GD*(T(i)+T(0.5));
      if(x<BLO||x>BHI) continue;
      for(int k=dims[1]-1;k>=0;--k){
        const int id=lat.cellId({i,k});
        if(id<0||fill[id]<=T(1e-6)) continue;
        h[binOf(x)]=lat.minCorner()[1]+GD*T(k+1)-(T(1)-fill[id])*GD; break;
      }
    }
    return h;
  };
  const Field before=probe();
  const auto t0=std::chrono::steady_clock::now();
  for(int s=0;s<steps;++s) vox.step(time/steps,cov,seed*100003u+1u+s);
  secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  Field h=probe();
  for(auto &kv:h){ auto it=before.find(kv.first); if(it!=before.end()) kv.second-=it->second; }
  return h;
}

// the band-averaged depth per run, then its mean and standard error: this is
// the ACCURACY half, which spread() throws away by construction
static void depth(const std::vector<Field>&runs,T &mean,T &sem){
  std::vector<T> m;
  for(const auto&f:runs){
    T sum=0; int n=0;
    for(const auto&kv:f){ sum+=kv.second; ++n; }
    m.push_back(n?sum/n:T(0));
  }
  T a=0; for(T x:m) a+=x; mean=m.empty()?T(0):a/m.size();
  T s=0; for(T x:m) s+=(x-mean)*(x-mean);
  sem=m.size()<2?T(0):std::sqrt(s/(m.size()-1))/std::sqrt((T)m.size());
}

// THREE readings of the SAME voxel state, to separate a probe artefact from a
// real difference:
//   top  -- first cell from the top with f > 1e-6, surface at top-(1-f)*Delta
//           (what every measurement so far has used)
//   half -- same formula but keyed on the METHOD's own material rule, f >= 0.5
//   vol  -- mass: column bottom + Delta * sum(f) over the whole column. No
//           choice of "which cell holds the surface" enters, so a sub-cell
//           reading error cannot survive here.
struct Probes { T top=0, half=0, vol=0; };

static Probes voxelProbes(ps::ChemicalMechanism<T> mech,T time,int steps,
                          size_t rays,unsigned seed){
  auto dom=makeDomain();
  auto topLS=dom->getLevelSets().back();
  auto deep=ls::SmartPointer<ls::Domain<T,D>>::New(topLS->getGrid());
  { T o[D]={0.,-36.}, n[D]={0.,1.};
    ls::MakeGeometry<T,D>(deep,ls::SmartPointer<ls::Plane<T,D>>::New(o,n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T,D>>> lss{deep};
  auto mm=ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::Si);
  for(size_t l=0;l<dom->getLevelSets().size();++l){
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto cellSet=viennacore::SmartPointer<cs::DenseCellSet<T,D>>::New();
  cellSet->setCellSetPosition(true);
  cellSet->setCoverMaterial((int)ps::Material::GAS);
  cellSet->fromLevelSets(lss,mm,BLANKET?T(10):MASKH+4.);
  cs::LatticeMap<T,D> lat(*cellSet);
  const auto &mid=*cellSet->getScalarData("Material");
  std::vector<T> fill(cellSet->getNumberOfCells(),T(0));
  std::vector<int> material(cellSet->getNumberOfCells());
  for(size_t c=0;c<fill.size();++c){
    material[c]=(int)mid[c];
    fill[c]= material[c]==(int)ps::Material::GAS?T(0):T(1);
  }
  ps::VoxelChemistry<T,D> vox(mech,lat,fill,material);
  vox.setRaysPerCell(rays);
  vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
  auto cov=vox.makeCoverages();
  const auto &dims=lat.dims();
  auto read=[&](){
    Probes p; int n=0;
    for(int i=0;i<dims[0];++i){
      const T x=lat.minCorner()[0]+GD*(T(i)+T(0.5));
      if(x<BLO||x>BHI) continue;
      bool gotTop=false, gotHalf=false; T sum=0;
      for(int k=dims[1]-1;k>=0;--k){
        const int id=lat.cellId({i,k});
        if(id<0) continue;
        const T f=fill[id];
        sum+=f;                                   // mass, whole column
        if(!gotTop && f>T(1e-6)){
          p.top+=lat.minCorner()[1]+GD*T(k+1)-(T(1)-f)*GD; gotTop=true; }
        if(!gotHalf && f>=T(0.5)){
          p.half+=lat.minCorner()[1]+GD*T(k+1)-(T(1)-f)*GD; gotHalf=true; }
      }
      p.vol+=lat.minCorner()[1]+GD*sum;
      ++n;
    }
    if(n){ p.top/=n; p.half/=n; p.vol/=n; }
    return p;
  };
  const Probes b=read();
  for(int s=0;s<steps;++s) vox.step(time/steps,cov,seed*100003u+1u+s);
  const Probes a=read();
  Probes d; d.top=a.top-b.top; d.half=a.half-b.half; d.vol=a.vol-b.vol;
  return d;
}

// spread over seeds: per bin the std, then the RMS over bins; plus the std of
// the band mean (the scalar the comparison actually reports)
static void spread(const std::vector<Field>&runs,T &sigField,T &sigScalar){
  std::map<int,std::vector<T>> byBin;
  std::vector<T> means;
  for(const auto&f:runs){
    T sum=0; int n=0;
    for(const auto&kv:f){ byBin[kv.first].push_back(kv.second); sum+=kv.second; ++n; }
    means.push_back(n?sum/n:T(0));
  }
  auto sd=[](const std::vector<T>&v){
    if(v.size()<2) return T(0);
    T m=0; for(T x:v) m+=x; m/=v.size();
    T s=0; for(T x:v) s+=(x-m)*(x-m);
    return std::sqrt(s/(v.size()-1));
  };
  T acc=0; int nb=0;
  for(const auto&kv:byBin){ if(kv.second.size()==runs.size()){ const T s=sd(kv.second); acc+=s*s; ++nb; } }
  sigField=nb?std::sqrt(acc/nb):T(0);
  sigScalar=sd(means);
}

// ---------------------------------------------------------------------------
// Generic, fully parameterised run: everything the hardcoded modes below do,
// driven from the command line so a new experiment is a shell loop, not an
// edit and a rebuild.
//
//   voxelNoise run geom=blanket delta=1 rays=400 cells=1 K=5 arm=both
//   voxelNoise run geom=trench  delta=1 rays=400 nm=10   K=15 steps=50
//
// keys: geom=blanket|trench  delta  rays  K  arm=ls|voxel|both
//       cells=<n>  (depth in CELLS)   or   nm=<x> (depth in nm)
//       steps=<n>|auto       band=lo,hi
// ---------------------------------------------------------------------------
static std::map<std::string,std::string> parseArgs(int argc,char**argv,int from){
  std::map<std::string,std::string> kv;
  for(int a=from;a<argc;++a){
    const std::string t=argv[a];
    const auto eq=t.find('=');
    if(eq!=std::string::npos) kv[t.substr(0,eq)]=t.substr(eq+1);
  }
  return kv;
}
static std::string argStr(const std::map<std::string,std::string>&kv,
                          const std::string&k,const std::string&dflt){
  auto it=kv.find(k); return it==kv.end()?dflt:it->second;
}
static double argNum(const std::map<std::string,std::string>&kv,
                     const std::string&k,double dflt){
  auto it=kv.find(k); return it==kv.end()?dflt:std::atof(it->second.c_str());
}

int main(int argc,char**argv){
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm"); ps::units::Time::setUnit("s");
  // mechanism is selectable: `mech=/path/file.json` or a bare name resolved
  // in the mechanism dir, so a chemistry variant (e.g. the ion channel
  // removed) is a data change rather than another rebuild
  std::string mechFile=std::string(VIENNAPS_MECHANISM_DIR)+"/sf6o2.mechanism.json";
  for(int a=1;a<argc;++a){
    const std::string t=argv[a];
    if(t.rfind("mech=",0)==0){
      const std::string v=t.substr(5);
      mechFile = v.find('/')==std::string::npos
          ? std::string(VIENNAPS_MECHANISM_DIR)+"/"+v : v;
    }
  }
  auto mech=ps::readChemicalMechanism<T>(mechFile);
  const T rate=47.74, time=5.0*2.0/rate;   // ~10 nm of etch, fixed physically
  const int K=15;
  const T rel=1.0/std::sqrt(2.0*(K-1));
  const std::string mode = argc>1 ? argv[1] : "budget";

  if(mode=="budget"){
    GD=1.0;
    std::cout<<"arm      budget      wall[s]   sigma_field[nm]  sigma_mean[nm]   +-\n";
    for(unsigned rays : {25u,50u,100u,200u,400u,800u}){
      std::vector<Field> runs; double tsum=0,t;
      for(int k=0;k<K;++k){ runs.push_back(lsArm(mech,time,rays,1000u+k,t)); tsum+=t; }
      T sf,sm; spread(runs,sf,sm);
      std::cout<<"levelset "<<std::setw(8)<<rays<<std::fixed<<std::setprecision(3)
               <<std::setw(11)<<tsum/K<<std::setw(15)<<sf<<std::setw(16)<<sm
               <<std::setw(9)<<sf*rel<<"\n"<<std::flush;
    }
    for(size_t rays : {25ul,50ul,100ul,200ul,400ul,800ul}){
      std::vector<Field> runs; double tsum=0,t;
      for(int k=0;k<K;++k){ runs.push_back(voxelArm(mech,time,50,rays,1000u+k,t)); tsum+=t; }
      T sf,sm; spread(runs,sf,sm);
      std::cout<<"voxel    "<<std::setw(8)<<rays<<std::fixed<<std::setprecision(3)
               <<std::setw(11)<<tsum/K<<std::setw(15)<<sf<<std::setw(16)<<sm
               <<std::setw(9)<<sf*rel<<"\n"<<std::flush;
    }
    return 0;
  }

  if(mode=="run"){
    const auto kv=parseArgs(argc,argv,2);
    const std::string geom=argStr(kv,"geom","trench");
    const std::string arm =argStr(kv,"arm","both");
    BLANKET = (geom=="blanket");
    INITCOV = argNum(kv,"init",0.0)!=0.0;
    COSPOW= argNum(kv,"cos",1.0);
    W     = argNum(kv,"W",40.0);
    MASKH = argNum(kv,"mask",30.0);
    GD      = argNum(kv,"delta",1.0);
    const unsigned rays=(unsigned)argNum(kv,"rays",400);
    const int KR=(int)argNum(kv,"K",5);
    if(BLANKET){ BLO=-20.; BHI=20.; }
    else { BLO=-0.35*W/2; BHI=0.35*W/2; }
    if(kv.count("band")){
      const std::string b=kv.at("band"); const auto c=b.find(',');
      if(c!=std::string::npos){ BLO=std::atof(b.substr(0,c).c_str());
                                BHI=std::atof(b.substr(c+1).c_str()); }
    }
    // depth: cells= (in grid cells) or nm= (absolute); default the 10 nm dose
    const auto gam=mech.sourceFluxes(ps::Material::Si);
    const auto kc=mech.rateConstantsFor(ps::Material::Si);
    std::vector<T> th(mech.coverageNames.size(),T(0));
    mech.solveCoverages(gam,kc,th);
    const T analytic=mech.growthRate(gam,kc,th,ps::Material::Si);
    T targetNm;
    if(kv.count("cells"))   targetNm=argNum(kv,"cells",1.0)*GD;
    else if(kv.count("nm")) targetNm=argNum(kv,"nm",10.0);
    else                    targetNm=std::abs(analytic)*time;
    const T tt=targetNm/std::abs(analytic);
    const std::string st=argStr(kv,"steps","auto");
    const int steps = st=="auto"
        ? (int)std::max(4L,std::lround(5.0*targetNm/GD))
        : (int)std::atol(st.c_str());
    std::cout<<"# geom="<<geom<<" delta="<<GD<<" rays="<<rays<<" K="<<KR
             <<" target="<<targetNm<<"nm ("<<targetNm/GD<<" cells) steps="
             <<steps<<" time="<<tt<<"s analytic="<<analytic<<"nm/s\n";
    std::cout<<"arm      delta   rays  elements    depth[nm]     sem"
               "   sigma_field   depth[cells]   vs_target\n";
    auto emit=[&](const char*nm,T mu,T se,T sf,size_t el){
      std::cout<<std::left<<std::setw(9)<<nm<<std::right<<std::fixed
               <<std::setprecision(2)<<std::setw(6)<<GD<<std::setw(7)<<rays
               <<std::setw(10)<<el<<std::setprecision(4)<<std::setw(13)<<mu
               <<std::setw(8)<<se<<std::setw(14)<<sf<<std::setw(15)<<mu/GD
               <<std::setprecision(2)<<std::setw(11)
               <<100*(mu/-targetNm-1)<<"%\n"<<std::flush;
    };
    if(arm=="ls"||arm=="both"){
      std::vector<Field> runs; double t; size_t el=0;
      for(int k=0;k<KR;++k) runs.push_back(lsArm(mech,tt,rays,1000u+k,t,&el));
      T mu,se,sf,sm; depth(runs,mu,se); spread(runs,sf,sm);
      emit("levelset",mu,se,sf,el);
    }
    if(arm=="voxel"||arm=="both"){
      std::vector<Field> runs; double t; size_t el=0;
      for(int k=0;k<KR;++k)
        runs.push_back(voxelArm(mech,tt,steps,rays,1000u+k,t,&el));
      T mu,se,sf,sm; depth(runs,mu,se); spread(runs,sf,sm);
      emit("voxel",mu,se,sf,el);
      if(INITCOV) std::cout<<"#   coverage pre-convergence: "<<INITSWEEPS
                           <<" sweeps\n";
    }
    return 0;
  }

  if(mode=="onecell"){
    // Etch an exact whole number of CELLS on a flat wafer. At exactly one
    // cell the surface lands back on a cell boundary, so partial-cell
    // handling has nowhere to hide. The 0.5 and 2.0 points are the control:
    // a constant relative rate error gives the same percentage at all three,
    // a cell-crossing artefact does not.
    BLANKET=true; BLO=-20.; BHI=20.;
    const auto gam=mech.sourceFluxes(ps::Material::Si);
    const auto kc=mech.rateConstantsFor(ps::Material::Si);
    std::vector<T> th(mech.coverageNames.size(),T(0));
    mech.solveCoverages(gam,kc,th);
    const T analytic=mech.growthRate(gam,kc,th,ps::Material::Si);
    std::cout<<"analytic rate "<<std::scientific<<std::setprecision(4)
             <<analytic<<" nm/s\n";
    std::cout<<"arm      delta  cells   target[nm]    depth[nm]     sem"
               "   depth[cells]   error\n";
    const int KC=5;
    for(T d : {2.0,1.0,0.5}){
      GD=d;
      for(T cells : {0.5,1.0,2.0}){
        const T target=cells*d;                 // nm to remove
        const T tt=target/std::abs(analytic);   // exact time for it
        const int steps=(int)std::max(4L,std::lround(cells*5.0));
        { std::vector<Field> runs; double t;
          for(int k=0;k<KC;++k) runs.push_back(lsArm(mech,tt,400u,1000u+k,t));
          T mu,se; depth(runs,mu,se);
          std::cout<<"levelset "<<std::fixed<<std::setprecision(2)<<std::setw(5)<<d
                   <<std::setw(7)<<cells<<std::setprecision(4)<<std::setw(12)<<-target
                   <<std::setw(13)<<mu<<std::setw(8)<<se
                   <<std::setw(14)<<mu/d<<std::setprecision(2)
                   <<std::setw(8)<<100*(mu/-target-1)<<"%\n"<<std::flush; }
        { std::vector<Field> runs; double t;
          for(int k=0;k<KC;++k) runs.push_back(voxelArm(mech,tt,steps,400ul,1000u+k,t));
          T mu,se; depth(runs,mu,se);
          std::cout<<"voxel    "<<std::fixed<<std::setprecision(2)<<std::setw(5)<<d
                   <<std::setw(7)<<cells<<std::setprecision(4)<<std::setw(12)<<-target
                   <<std::setw(13)<<mu<<std::setw(8)<<se
                   <<std::setw(14)<<mu/d<<std::setprecision(2)
                   <<std::setw(8)<<100*(mu/-target-1)<<"%\n"<<std::flush; }
      }
    }
    return 0;
  }

  if(mode=="zero"){
    // Null test: build the geometry, put it through the whole flow with
    // ZERO process time, read the surface again. Anything that moves here
    // moved without any etching -- initialisation, the probe, or a pass in
    // the step that runs regardless of dt. ABSOLUTE positions, not
    // differences, so the two arms can also be compared at t = 0.
    BLANKET=true; BLO=-20.; BHI=20.;
    auto bandMean=[](const Field &f){
      T sum=0; int n=0;
      for(const auto&kv:f){ sum+=kv.second; ++n; }
      return n?sum/n:T(0);
    };
    std::cout<<"arm      delta    h_built     h_after_0s      drift\n";
    for(T d : {2.0,1.0,0.5}){
      GD=d;
      const int steps=(int)std::lround(50.0/d);
      { // ---- level set: Process with duration 0
        auto dom=makeDomain();
        const T h0=bandMean(lsField(dom));
        auto model=ps::SmartPointer<ps::SurfaceChemistry<T,D>>::New(mech);
        ps::Process<T,D> proc(dom,model,T(1e-9));
        proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
        ps::RayTracingParameters rt; rt.raysPerPoint=400;
        rt.useRandomSeeds=false; rt.rngSeed=1000u; proc.setParameters(rt);
        ps::CoverageParameters cv; cv.tolerance=1e-6; cv.maxIterations=40;
        proc.setParameters(cv);
        proc.apply();
        const T h1=bandMean(lsField(dom));
        std::cout<<"levelset "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
                 <<std::setprecision(6)<<std::setw(12)<<h0<<std::setw(14)<<h1
                 <<std::setw(12)<<h1-h0<<"\n"<<std::flush;
      }
      { // ---- voxel: the same number of steps, each with dt = 0
        auto dom=makeDomain();
        auto topLS=dom->getLevelSets().back();
        auto deep=ls::SmartPointer<ls::Domain<T,D>>::New(topLS->getGrid());
        { T o[D]={0.,-36.}, n[D]={0.,1.};
          ls::MakeGeometry<T,D>(deep,ls::SmartPointer<ls::Plane<T,D>>::New(o,n)).apply(); }
        std::vector<ls::SmartPointer<ls::Domain<T,D>>> lss{deep};
        auto mm=ls::SmartPointer<ls::MaterialMap>::New();
        mm->insertNextMaterial((int)ps::Material::Si);
        for(size_t l=0;l<dom->getLevelSets().size();++l){
          lss.push_back(dom->getLevelSets()[l]);
          mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
        }
        auto cellSet=viennacore::SmartPointer<cs::DenseCellSet<T,D>>::New();
        cellSet->setCellSetPosition(true);
        cellSet->setCoverMaterial((int)ps::Material::GAS);
        cellSet->fromLevelSets(lss,mm,T(10));
        cs::LatticeMap<T,D> lat(*cellSet);
        const auto &mid=*cellSet->getScalarData("Material");
        std::vector<T> fill(cellSet->getNumberOfCells(),T(0));
        std::vector<int> material(cellSet->getNumberOfCells());
        for(size_t c=0;c<fill.size();++c){
          material[c]=(int)mid[c];
          fill[c]= material[c]==(int)ps::Material::GAS?T(0):T(1);
        }
        ps::VoxelChemistry<T,D> vox(mech,lat,fill,material);
        vox.setRaysPerCell(400);
        vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
        auto cov=vox.makeCoverages();
        const auto &dims=lat.dims();
        auto probe=[&](){
          Field h;
          for(int i=0;i<dims[0];++i){
            const T x=lat.minCorner()[0]+GD*(T(i)+T(0.5));
            if(x<BLO||x>BHI) continue;
            for(int k=dims[1]-1;k>=0;--k){
              const int id=lat.cellId({i,k});
              if(id<0||fill[id]<=T(1e-6)) continue;
              h[binOf(x)]=lat.minCorner()[1]+GD*T(k+1)-(T(1)-fill[id])*GD; break;
            }
          }
          return h;
        };
        const T h0=bandMean(probe());
        for(int st=0;st<steps;++st) vox.step(T(0),cov,1u+st);
        const T h1=bandMean(probe());
        std::cout<<"voxel dt0"<<std::fixed<<std::setprecision(2)<<std::setw(5)<<d
                 <<std::setprecision(6)<<std::setw(12)<<h0<<std::setw(14)<<h1
                 <<std::setw(12)<<h1-h0<<"\n"<<std::flush;
        for(int st=0;st<steps;++st) vox.step(T(1e-9)/steps,cov,1u+st);
        const T h2=bandMean(probe());
        std::cout<<"voxel tny"<<std::fixed<<std::setprecision(2)<<std::setw(5)<<d
                 <<std::setprecision(6)<<std::setw(12)<<h1<<std::setw(14)<<h2
                 <<std::setw(12)<<h2-h1<<"\n"<<std::flush;
      }
    }
    return 0;
  }

  if(mode=="blanket"){
    // No mask, flat surface, same mechanism and budgets. Analytic rate from
    // the mechanism itself, so both arms are measured against physics rather
    // than against each other.
    BLANKET=true; BLO=-20.; BHI=20.;
    const auto gam=mech.sourceFluxes(ps::Material::Si);
    const auto kc=mech.rateConstantsFor(ps::Material::Si);
    std::vector<T> th(mech.coverageNames.size(),T(0));
    mech.solveCoverages(gam,kc,th);
    const T analytic=mech.growthRate(gam,kc,th,ps::Material::Si);
    const T expect=analytic*time;
    std::cout<<"analytic rate "<<std::scientific<<std::setprecision(4)<<analytic
             <<" nm/s, time "<<time<<" s, expected depth "<<std::fixed
             <<std::setprecision(4)<<expect<<" nm\n";
    std::cout<<"arm      delta   rays    depth[nm]     sem   vs analytic\n";
    const int KB=5;
    for(T d : {2.0,1.0,0.5}){
      GD=d;
      const int steps=(int)std::lround(50.0/d);
      { std::vector<Field> runs; double t;
        for(int k=0;k<KB;++k) runs.push_back(lsArm(mech,time,400u,1000u+k,t));
        T mu,se; depth(runs,mu,se);
        std::cout<<"levelset "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
                 <<std::setw(7)<<400<<std::setprecision(4)<<std::setw(12)<<mu
                 <<std::setw(8)<<se<<std::setprecision(2)
                 <<std::setw(11)<<100*(mu/expect-1)<<"%\n"<<std::flush; }
      { std::vector<Field> runs; double t;
        for(int k=0;k<KB;++k) runs.push_back(voxelArm(mech,time,steps,400ul,1000u+k,t));
        T mu,se; depth(runs,mu,se);
        std::cout<<"voxel    "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
                 <<std::setw(7)<<400<<std::setprecision(4)<<std::setw(12)<<mu
                 <<std::setw(8)<<se<<std::setprecision(2)
                 <<std::setw(11)<<100*(mu/expect-1)<<"%\n"<<std::flush; }
    }
    return 0;
  }

  if(mode=="probe"){
    // Is the 0.17 nm a reading artefact? Same state, three definitions.
    std::cout<<"delta  rays   K   depth_top   depth_half    depth_vol   "
               "top-vol   half-vol\n";
    for(T d : {2.0,1.0,0.5}){
      GD=d;
      const int steps=(int)std::lround(50.0/d);
      const int KP=5;
      T st=0,sh=0,sv=0;
      for(int k=0;k<KP;++k){
        const Probes p=voxelProbes(mech,time,steps,400ul,1000u+k);
        st+=p.top; sh+=p.half; sv+=p.vol;
      }
      st/=KP; sh/=KP; sv/=KP;
      std::cout<<std::fixed<<std::setprecision(2)<<std::setw(5)<<d
               <<std::setw(6)<<400<<std::setw(4)<<KP<<std::setprecision(4)
               <<std::setw(12)<<st<<std::setw(13)<<sh<<std::setw(13)<<sv
               <<std::setw(10)<<st-sv<<std::setw(10)<<sh-sv<<"\n"<<std::flush;
    }
    return 0;
  }

  if(mode=="matrix"){
    // The question this answers: at which (rays per surface element, Delta)
    // do the two arms give the SAME answer? Delta sets the systematic depth,
    // the ray budget sets the scatter around it, so both are reported and the
    // two are matched independently.
    std::cout<<"arm      delta   rays/el   elements   totalrays    "
               "depth[nm]     sem   sigma_field[nm]  sigma_mean[nm]   +-\n";
    for(T d : {2.0,1.0,0.5}){
      GD=d;
      const int steps=(int)std::lround(50.0/d);
      for(unsigned rays : {25u,100u,400u,1600u}){
        { std::vector<Field> runs; double tsum=0,t; size_t nodes=0;
          for(int k=0;k<K;++k){ runs.push_back(lsArm(mech,time,rays,1000u+k,t,&nodes)); tsum+=t; }
          T sf,sm,mu,se; spread(runs,sf,sm); depth(runs,mu,se);
          std::cout<<"levelset "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
                   <<std::setw(10)<<rays<<std::setw(11)<<nodes
                   <<std::setw(12)<<(size_t)nodes*rays<<std::setprecision(3)
                   <<std::setw(12)<<mu<<std::setw(8)<<se
                   <<std::setw(15)<<sf<<std::setw(16)<<sm
                   <<std::setw(9)<<sf*rel<<"\n"<<std::flush; }
        { std::vector<Field> runs; double tsum=0,t; size_t cells=0;
          for(int k=0;k<K;++k){ runs.push_back(voxelArm(mech,time,steps,rays,1000u+k,t,&cells)); tsum+=t; }
          T sf,sm,mu,se; spread(runs,sf,sm); depth(runs,mu,se);
          std::cout<<"voxel    "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
                   <<std::setw(10)<<rays<<std::setw(11)<<cells
                   <<std::setw(12)<<(size_t)cells*rays<<std::setprecision(3)
                   <<std::setw(12)<<mu<<std::setw(8)<<se
                   <<std::setw(15)<<sf<<std::setw(16)<<sm
                   <<std::setw(9)<<sf*rel<<"\n"<<std::flush; }
      }
    }
    return 0;
  }

  // Delta sweep at the plateau budgets found above: if the floors are set by
  // discretisation they must fall with Delta; if they are an iteration or
  // solver artefact they will not.
  std::cout<<"arm      delta   budget      wall[s]   sigma_field[nm]  sigma_mean[nm]   +-\n";
  for(T d : {2.0,1.0,0.5}){
    GD=d;
    { std::vector<Field> runs; double tsum=0,t;
      for(int k=0;k<K;++k){ runs.push_back(lsArm(mech,time,800u,1000u+k,t)); tsum+=t; }
      T sf,sm; spread(runs,sf,sm);
      std::cout<<"levelset "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
               <<std::setw(9)<<800<<std::setprecision(3)
               <<std::setw(11)<<tsum/K<<std::setw(15)<<sf<<std::setw(16)<<sm
               <<std::setw(9)<<sf*rel<<"\n"<<std::flush; }
    { // steps scale with 1/delta so the advance per step stays a fixed
      // fraction of a cell as the grid is refined
      const int steps=(int)std::lround(50.0/d);
      std::vector<Field> runs; double tsum=0,t;
      for(int k=0;k<K;++k){ runs.push_back(voxelArm(mech,time,steps,800ul,1000u+k,t)); tsum+=t; }
      T sf,sm; spread(runs,sf,sm);
      std::cout<<"voxel    "<<std::fixed<<std::setprecision(2)<<std::setw(6)<<d
               <<std::setw(9)<<800<<std::setprecision(3)
               <<std::setw(11)<<tsum/K<<std::setw(15)<<sf<<std::setw(16)<<sm
               <<std::setw(9)<<sf*rel<<"\n"<<std::flush; }
  }
}
