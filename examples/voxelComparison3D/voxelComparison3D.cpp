// The two arms in THREE dimensions: same trench, same mechanism file, level
// set against voxels. Everything below is the 2D comparison re-run with D = 3
// -- the components are dimension-templated, but a template argument is a
// claim, not a verification, and the full chain has never executed in 3D.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>
#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <process/psProcess.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>
namespace ps=viennaps; namespace cs=viennacs; namespace ls=viennals;
using T=double; constexpr int D=3;

struct Profile { T field=0, bottom=0, cov=0; };
static const T W=20., DEPTH=30., GD=1.0;
static const T FLO=-0.45*4*W, FHI=-0.30*4*W, BLO=-0.35*W/2, BHI=0.35*W/2;

Profile levelSetArm(ps::ChemicalMechanism<T> mech, T time, unsigned rays,
                    T maskHeight = 0, bool maskOnly = false){
  const T trenchD = maskOnly ? T(0) : DEPTH;
  auto dom=ps::SmartPointer<ps::Domain<T,D>>::New(GD,T(4)*W,T(4)*W);
  ps::MakeTrench<T,D>(dom,W,trenchD,T(0),maskHeight,T(0),false,
                      ps::Material::Si,ps::Material::Mask).apply();
  auto bounds=[&](){
    auto mesh=ps::SmartPointer<ls::Mesh<T>>::New();
    ls::ToSurfaceMesh<T,D>(dom->getLevelSets().back(),mesh).apply();
    T f=std::numeric_limits<T>::lowest(), b=std::numeric_limits<T>::lowest();
    for(const auto&n:mesh->getNodes()){
      if(n[0]>=FLO&&n[0]<=FHI) f=std::max(f,n[D-1]);
      if(n[0]>=BLO&&n[0]<=BHI) b=std::max(b,n[D-1]);
    }
    return std::pair<T,T>{f,b};
  };
  const auto before=bounds();
  dom->saveSurfaceMesh(mech.name+"_3d_ls_initial.vtp");
  auto model=ps::SmartPointer<ps::SurfaceChemistry<T,D>>::New(mech);
  ps::Process<T,D> p(dom,model,time);
  p.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
  ps::RayTracingParameters rt; rt.raysPerPoint=rays; p.setParameters(rt);
  ps::CoverageParameters cv; cv.tolerance=1e-6; cv.maxIterations=40; p.setParameters(cv);
  p.apply();
  const auto after=bounds();
  dom->saveSurfaceMesh(mech.name+"_3d_ls_final.vtp");
  Profile r; r.field=after.first-before.first; r.bottom=after.second-before.second;
  r.cov=std::abs(r.field)>1e-12?r.bottom/r.field:0; return r;
}

Profile voxelArm(ps::ChemicalMechanism<T> mech, T time, int steps, size_t rays,
                 cs::NormalEstimator est, T maskHeight = 0, bool maskOnly = false){
  const T trenchD = maskOnly ? T(0) : DEPTH;
  auto dom=ps::SmartPointer<ps::Domain<T,D>>::New(GD,T(4)*W,T(4)*W);
  ps::MakeTrench<T,D>(dom,W,trenchD,T(0),maskHeight,T(0),false,
                      ps::Material::Si,ps::Material::Mask).apply();
  auto topLS=dom->getLevelSets().back();
  auto deep=ls::SmartPointer<ls::Domain<T,D>>::New(topLS->getGrid());
  { T o[D]={0.,0.,-(DEPTH+T(6))}, n[D]={0.,0.,1.};
    ls::MakeGeometry<T,D>(deep,ls::SmartPointer<ls::Plane<T,D>>::New(o,n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T,D>>> lss{deep};
  auto matMap=ls::SmartPointer<ls::MaterialMap>::New();
  matMap->insertNextMaterial((int)ps::Material::Si);
  for(size_t l=0;l<dom->getLevelSets().size();++l){
    lss.push_back(dom->getLevelSets()[l]);
    matMap->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto cellSet=viennacore::SmartPointer<cs::DenseCellSet<T,D>>::New();
  cellSet->setCellSetPosition(true);
  cellSet->setCoverMaterial((int)ps::Material::GAS);
  cellSet->fromLevelSets(lss,matMap,maskHeight+T(4)); // cover must clear the mask

  cs::LatticeMap<T,D> lat(*cellSet);
  const auto&mid=*cellSet->getScalarData("Material");
  std::vector<T> fill(cellSet->getNumberOfCells(),T(0));
  std::vector<int> material(cellSet->getNumberOfCells());
  for(size_t c=0;c<fill.size();++c){
    const bool solid=(int)mid[c]!=(int)ps::Material::GAS;
    fill[c]=solid?T(1):T(0);
    material[c]=solid?(int)ps::Material::Si:(int)ps::Material::GAS;
  }
  ps::VoxelChemistry<T,D> vox(mech,lat,fill,material);
  vox.setNormalEstimator(est);
  vox.setRaysPerStep(rays);
  auto cov=vox.makeCoverages();
  const auto&dims=lat.dims();
  auto surf=[&](T lo,T hi){
    T sum=0; int n=0;
    for(int i=0;i<dims[0];++i){
      const T x=lat.minCorner()[0]+GD*(T(i)+T(0.5));
      if(x<lo||x>hi) continue;
      for(int j=0;j<dims[1];++j){
        for(int k=dims[2]-1;k>=0;--k){
          const int id=lat.cellId({i,j,k});
          if(id<0||fill[id]<=T(1e-6)) continue;
          sum+=lat.minCorner()[2]+GD*T(k+1)-(T(1)-fill[id])*GD; ++n; break;
        }
      }
    }
    return n?sum/n:T(0);
  };
  auto writeCells=[&](const std::string&f){
    auto &ff=*cellSet->getFillingFractions();
    auto &mm=*cellSet->getScalarData("Material");
    const auto &labels=vox.materials(); // the EVOLVED labels, as in 2D
    for(size_t c=0;c<fill.size();++c){ ff[c]=fill[c]; mm[c]=(T)labels[c]; }
    cellSet->writeVTU(f);
  };
  writeCells(mech.name+"_3d_voxel_initial.vtu");
  const T f0=surf(FLO,FHI), b0=surf(BLO,BHI);
  for(int s=0;s<steps;++s) vox.step(time/steps,cov,1+s);
  writeCells(mech.name+"_3d_voxel_final.vtu");
  Profile r; r.field=surf(FLO,FHI)-f0; r.bottom=surf(BLO,BHI)-b0;
  r.cov=std::abs(r.field)>1e-12?r.bottom/r.field:0; return r;
}

#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

int main(int argc,char**argv){
  ps::Logger::setLogLevel(ps::LogLevel::WARNING);
  ps::units::Length::setUnit("nm"); ps::units::Time::setUnit("s");
  std::vector<std::string> files;
  if(argc>1) for(int a=1;a<argc;++a) files.emplace_back(argv[a]);
  else for(const char*m:{"silane","sf6o2"})
    files.emplace_back(std::string(VIENNAPS_MECHANISM_DIR)+"/"+m+".mechanism.json");
  for(const auto&file:files){
    auto mech=ps::readChemicalMechanism<T>(file);
    const auto gam=mech.sourceFluxes(ps::Material::Si);
    const auto k=mech.rateConstantsFor(ps::Material::Si);
    std::vector<T> th(mech.coverageNames.size(),T(0));
    mech.solveCoverages(gam,k,th);
    const T analytic=mech.growthRate(gam,k,th,ps::Material::Si);
    const T time=T(0.2)*W/2/std::abs(analytic); // 2 nm target
    std::cout<<"3D trench: "<<mech.name<<"  (analytic "<<std::scientific
             <<std::setprecision(3)<<analytic<<" nm/s)\n";
    auto row=[&](const char*t,const Profile&p){
      std::cout<<"  "<<std::left<<std::setw(22)<<t<<std::right<<std::fixed
               <<std::setprecision(3)<<std::setw(9)<<p.field<<std::setw(10)
               <<p.bottom;
      // floor over a static mask top is not a step coverage; print nothing
      if(std::abs(p.field)>T(0.05)) std::cout<<std::setw(9)<<p.cov;
      else std::cout<<std::setw(9)<<"-";
      std::cout<<"\n"; };
    const T maskH = analytic<0 ? T(15) : T(0);
    const bool maskOnly = maskH > 0; // flat substrate; the mask is the opening
    const T timeR = maskOnly ? 5*time : time; // dig ~10 nm of real topography
    const auto lsr=levelSetArm(mech,timeR,200,maskH,maskOnly);
    row(maskH>0?"level set (mask/floor)":"level set",lsr);
    const auto vy=voxelArm(mech,timeR,maskOnly?50:10,500000,
                           cs::NormalEstimator::FillGradientYoungs,maskH,maskOnly);
    row(maskH>0?"voxel, Youngs (mask/floor)":"voxel, Youngs",vy);
    if(maskH>0)
      // With a mask the field column IS the mask top, which barely moves, so
      // a floor-over-field ratio divides by nearly zero and means nothing:
      // the floor against the reference is the comparison.
      std::cout<<"  floor, voxel against level set: "<<std::setprecision(1)
               <<100*(vy.bottom/lsr.bottom-1)<<"%\n\n";
    else
      std::cout<<"  step coverage, voxel against level set: "<<std::setprecision(1)
               <<100*(vy.cov/lsr.cov-1)<<"%   field vs analytic: LS "
               <<100*(lsr.field/(analytic*time)-1)<<"%, voxel "
               <<100*(vy.field/(analytic*time)-1)<<"%\n\n";
  }
}
