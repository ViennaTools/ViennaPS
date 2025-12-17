#pragma once

#ifdef VIENNALS_VTK_RENDERING

#include <lsMaterialMap.hpp>
#include <lsMesh.hpp>
#include <lsWriteVisualizationMesh.hpp>

#include <vtkActor.h>
#include <vtkAutoInit.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkInteractorStyleImage.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#ifndef VIENNALS_VTK_MODULE_INIT
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);
VTK_MODULE_INIT(vtkRenderingUI);
#endif

namespace viennaps {
template <typename NumericType, int D> class VTKRenderWindow;

enum class RenderMode { SURFACE, INTERFACE, VOLUME };
} // namespace viennaps

// Forward declaration - full definition after VTKRenderWindow
class Custom3DInteractorStyle;
class Custom2DInteractorStyle;

namespace viennaps {

// forward declaration of Domain
template <typename T, int D> class Domain;

template <typename T, int D> class VTKRenderWindow {
public:
  VTKRenderWindow() { initialize(); }
  VTKRenderWindow(SmartPointer<Domain<T, D>> passedDomain)
      : domain(passedDomain) {
    initialize();
  }

  ~VTKRenderWindow();

  void setBackgroundColor(const std::array<double, 3> &color) {
    backgroundColor = color;
    if (renderer) {
      renderer->SetBackground(backgroundColor.data());
    }
  }

  void setRenderMode(RenderMode mode) { renderMode = mode; }

  void render() {
    updateDisplay();
    interactor->Initialize();
    interactor->Start();
  }

  // Update the display without starting a new event loop (for use during
  // interaction)
  void updateDisplay() {
    updateRenderer();
    renderWindow->Render();
  }

private:
  void initialize();

  void updateRenderer() {
    if (!domain) {
      VIENNACORE_LOG_WARNING("No domain set for rendering.");
      return;
    }

    // Clear previous actors
    renderer->RemoveAllViewProps();

    switch (renderMode) {
    case RenderMode::SURFACE: {
      auto surfaceMesh = domain->getSurfaceMesh();
      updatePolyData(surfaceMesh);
      break;
    }
    case RenderMode::INTERFACE: {
      auto interfaceMesh = domain->getSurfaceMesh(true);
      updatePolyData(interfaceMesh);
      break;
    }
    case RenderMode::VOLUME: {
      viennals::WriteVisualizationMesh<T, D> writer;
      auto matMap = domain->getMaterialMap();
      auto levelSets = domain->getLevelSets();
      int minMatId = std::numeric_limits<int>::max();
      int maxMatId = std::numeric_limits<int>::min();
      for (auto i{0u}; i < domain->getNumberOfLevelSets(); ++i) {
        writer.insertNextLevelSet(levelSets[i]);
        if (matMap) {
          int lsMinId = static_cast<int>(matMap->getMaterialAtIdx(i));
          int lsMaxId = static_cast<int>(matMap->getMaterialAtIdx(i));
          minMatId = std::min(minMatId, lsMinId);
          maxMatId = std::max(maxMatId, lsMaxId);
        }
      }
      writer.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
      writer.setWriteToFile(false);
      writer.apply();

      auto volumeMesh = writer.getVolumeMesh();
      updateVolumeMesh(volumeMesh, minMatId, maxMatId);
      break;
    }
    default:
      assert(false && "Unknown render mode.");
    }
  }

private:
  void updatePolyData(SmartPointer<viennals::Mesh<T>> mesh) {
    if (mesh == nullptr) {
      return;
    }

    auto polyData = vtkSmartPointer<vtkPolyData>::New();

    // Points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (const auto &node : mesh->getNodes()) {
      points->InsertNextPoint(node[0], node[1], node[2]);
    }
    polyData->SetPoints(points);

    // Vertices
    if (!mesh->vertices.empty()) {
      vtkSmartPointer<vtkCellArray> verts =
          vtkSmartPointer<vtkCellArray>::New();
      for (const auto &vertex : mesh->vertices) {
        verts->InsertNextCell(1);
        verts->InsertCellPoint(vertex[0]);
      }
      polyData->SetVerts(verts);
    }

    // Lines
    if (!mesh->lines.empty()) {
      vtkSmartPointer<vtkCellArray> lines =
          vtkSmartPointer<vtkCellArray>::New();
      for (const auto &line : mesh->lines) {
        lines->InsertNextCell(2);
        lines->InsertCellPoint(line[0]);
        lines->InsertCellPoint(line[1]);
      }
      polyData->SetLines(lines);
    }

    // Triangles
    if (!mesh->triangles.empty()) {
      if constexpr (D < 3) {
        VIENNACORE_LOG_WARNING("Adding triangles to a 2D mesh.");
      }

      vtkSmartPointer<vtkCellArray> polys =
          vtkSmartPointer<vtkCellArray>::New();
      for (const auto &triangle : mesh->triangles) {
        polys->InsertNextCell(3);
        polys->InsertCellPoint(triangle[0]);
        polys->InsertCellPoint(triangle[1]);
        polys->InsertCellPoint(triangle[2]);
      }
      polyData->SetPolys(polys);
    }

    // Material IDs as cell data
    auto materialIds = mesh->getCellData().getScalarData("MaterialIds", true);
    bool useMaterialIds =
        materialIds &&
        (materialIds->size() == mesh->lines.size() + mesh->triangles.size());
    int minId = std::numeric_limits<int>::max();
    int maxId = std::numeric_limits<int>::min();
    if (useMaterialIds) {
      vtkSmartPointer<vtkIntArray> matIdArray =
          vtkSmartPointer<vtkIntArray>::New();
      matIdArray->SetName("MaterialIds");
      for (const auto &id : *materialIds) {
        int mId = static_cast<int>(id);
        matIdArray->InsertNextValue(mId);
        minId = std::min(minId, mId);
        maxId = std::max(maxId, mId);
      }
      polyData->GetCellData()->AddArray(matIdArray);
      polyData->GetCellData()->SetActiveScalars("MaterialIds");
      VIENNACORE_LOG_DEBUG("Added MaterialIds array to cell data.");
    }

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    if (useMaterialIds) {
      mapper->SetScalarModeToUseCellData();
      mapper->ScalarVisibilityOn();
      mapper->SelectColorArray("MaterialIds");

      vtkSmartPointer<vtkLookupTable> lut =
          vtkSmartPointer<vtkLookupTable>::New();

      lut->SetNumberOfTableValues(256);
      lut->SetHueRange(0.667, 0.0); // blue → red
      lut->SetSaturationRange(1.0, 1.0);
      lut->SetValueRange(1.0, 1.0);
      lut->Build();

      mapper->SetLookupTable(lut);
      mapper->SetScalarRange(minId, maxId);
    }

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetLineWidth(3.0); // Thicker lines

    renderer->AddActor(actor);
    renderer->ResetCamera();
  }

  void updateVolumeMesh(vtkSmartPointer<vtkUnstructuredGrid> volumeMesh,
                        int minMatId = 0, int maxMatId = 100) {
    auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(volumeMesh);
    mapper->SetScalarModeToUseCellData();
    mapper->ScalarVisibilityOn();
    mapper->SelectColorArray("Material");

    vtkSmartPointer<vtkLookupTable> lut =
        vtkSmartPointer<vtkLookupTable>::New();

    lut->SetNumberOfTableValues(256);
    lut->SetHueRange(0.667, 0.0); // blue → red
    lut->SetSaturationRange(1.0, 1.0);
    lut->SetValueRange(1.0, 1.0);
    lut->Build();

    mapper->SetLookupTable(lut);
    mapper->SetScalarRange(minMatId, maxMatId);

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    renderer->AddActor(actor);
    renderer->ResetCamera();
  }

private:
  SmartPointer<Domain<T, D>> domain;

  vtkSmartPointer<vtkRenderer> renderer;
  vtkSmartPointer<vtkRenderWindow> renderWindow;
  vtkSmartPointer<vtkRenderWindowInteractor> interactor;

  std::array<double, 3> backgroundColor = {84.0 / 255, 89.0 / 255, 109.0 / 255};
  std::array<int, 2> windowSize = {800, 600};

  RenderMode renderMode = RenderMode::INTERFACE;
};

} // namespace viennaps

// Full definition of Custom3DInteractorStyle after VTKRenderWindow is complete
class Custom3DInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
  static Custom3DInteractorStyle *New();
  vtkTypeMacro(Custom3DInteractorStyle, vtkInteractorStyleTrackballCamera);

  virtual void OnChar() {
    if (Window == nullptr)
      return;

    vtkRenderWindowInteractor *rwi = this->Interactor;
    if (!rwi || rwi->GetDone())
      return;

    switch (rwi->GetKeyCode()) {
    case '1':
      Window->setRenderMode(viennaps::RenderMode::SURFACE);
      Window->updateDisplay();
      return;
    case '2':
      Window->setRenderMode(viennaps::RenderMode::INTERFACE);
      Window->updateDisplay();
      return;
    case '3':
      Window->setRenderMode(viennaps::RenderMode::VOLUME);
      Window->updateDisplay();
      return;
    }
  }

  void setRenderWindow(viennaps::VTKRenderWindow<double, 3> *window) {
    this->Window = window;
  }

private:
  viennaps::VTKRenderWindow<double, 3> *Window = nullptr;
};

vtkStandardNewMacro(Custom3DInteractorStyle);

class Custom2DInteractorStyle : public vtkInteractorStyleImage {
public:
  static Custom2DInteractorStyle *New();
  vtkTypeMacro(Custom2DInteractorStyle, vtkInteractorStyleImage);

  void OnLeftButtonDown() override { this->StartPan(); }

  void OnLeftButtonUp() override { this->EndPan(); }

  virtual void OnChar() {
    if (Window == nullptr)
      return;

    vtkRenderWindowInteractor *rwi = this->Interactor;
    if (!rwi || rwi->GetDone())
      return;

    switch (rwi->GetKeyCode()) {
    case '1':
      Window->setRenderMode(viennaps::RenderMode::SURFACE);
      Window->updateDisplay();
      return;
    case '2':
      Window->setRenderMode(viennaps::RenderMode::INTERFACE);
      Window->updateDisplay();
      return;
    case '3':
      Window->setRenderMode(viennaps::RenderMode::VOLUME);
      Window->updateDisplay();
      return;
    }
  }

  void setRenderWindow(viennaps::VTKRenderWindow<double, 2> *window) {
    this->Window = window;
  }

private:
  viennaps::VTKRenderWindow<double, 2> *Window = nullptr;
};

vtkStandardNewMacro(Custom2DInteractorStyle);

// VTKRenderWindow::initialize() implementation - defined after
// Custom3DInteractorStyle
namespace viennaps {

template <typename T, int D> VTKRenderWindow<T, D>::~VTKRenderWindow() {
  if (interactor) {
    if (auto *customStyle = Custom3DInteractorStyle::SafeDownCast(
            interactor->GetInteractorStyle())) {
      customStyle->setRenderWindow(nullptr);
    }
    interactor->SetInteractorStyle(nullptr);
    interactor->SetRenderWindow(nullptr);
  }
  if (renderWindow) {
    renderWindow->SetInteractor(nullptr);
    renderWindow->RemoveRenderer(renderer);
  }
}

template <typename T, int D> void VTKRenderWindow<T, D>::initialize() {
  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->SetBackground(backgroundColor.data());

  renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->SetWindowName("ViennaPS Render Window");
  renderWindow->SetSize(windowSize.data());
  renderWindow->AddRenderer(renderer);

  // Initialize interactor
  interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  if constexpr (D == 2) {
    vtkCamera *cam = renderer->GetActiveCamera();
    cam->ParallelProjectionOn();
    auto style = vtkSmartPointer<Custom2DInteractorStyle>::New();
    style->setRenderWindow(this);
    interactor->SetInteractorStyle(style);
  } else {
    auto style = vtkSmartPointer<Custom3DInteractorStyle>::New();
    style->setRenderWindow(this);
    interactor->SetInteractorStyle(style);
  }

  interactor->SetRenderWindow(renderWindow);
  renderWindow->SetInteractor(interactor);

  if (auto style = interactor->GetInteractorStyle()) {
    style->SetDefaultRenderer(renderer);
    style->SetCurrentRenderer(renderer);
  }
}
} // namespace viennaps

#endif // VIENNALS_VTK_RENDERING