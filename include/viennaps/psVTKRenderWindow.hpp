#pragma once

#ifdef VIENNALS_VTK_RENDERING

#include <lsMaterialMap.hpp>
#include <lsMesh.hpp>
#include <lsWriteVisualizationMesh.hpp>

#include "psUtil.hpp"

#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkInteractorStyleImage.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkPNGWriter.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#ifndef VIENNALS_VTK_MODULE_INIT
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);
VTK_MODULE_INIT(vtkRenderingUI);
#endif

// Forward declaration - full definition after VTKRenderWindow
class Custom3DInteractorStyle;
class Custom2DInteractorStyle;

namespace viennaps {
enum class RenderMode { SURFACE, INTERFACE, VOLUME };

// forward declaration of Domain
VIENNAPS_TEMPLATE_ND_FWD(NumericType, D) class Domain;

/// Lightweight VTK-based viewer for one or more ViennaPS domains.
///
/// Wraps a renderer, window, and interactor to visualize level-set derived
/// meshes either as surfaces, interfaces, or volume cells. Provides helpers to
/// manage camera presets, toggle overlays, and capture screenshots without the
/// caller dealing with raw VTK plumbing.
template <typename T, int D> class VTKRenderWindow {
public:
  VTKRenderWindow() { initialize(); }
  VTKRenderWindow(SmartPointer<Domain<T, D>> passedDomain) {
    initialize();
    insertNextDomain(passedDomain);
  }

  ~VTKRenderWindow() {
    if (interactor) {
      interactor->SetInteractorStyle(nullptr);
      interactor->SetRenderWindow(nullptr);
    }
    if (renderWindow) {
      renderWindow->SetInteractor(nullptr);
      renderWindow->RemoveRenderer(renderer);
    }
  }

  /// Queue a domain for rendering with an optional translation offset.
  void insertNextDomain(SmartPointer<Domain<T, D>> passedDomain,
                        const std::array<double, 3> &offset = {0.0, 0.0, 0.0}) {
    domains.push_back(passedDomain);
    domainOffsets.push_back(offset);
  }

  /// Adjust the translation offset of an already enqueued domain.
  void setDomainOffset(std::size_t domainIndex,
                       const std::array<double, 3> &offset) {
    if (domainIndex >= domainOffsets.size()) {
      VIENNACORE_LOG_ERROR("Domain index out of range.");
      return;
    }
    domainOffsets[domainIndex] = offset;
  }

  /// Set the renderer background color (RGB in [0,1]).
  void setBackgroundColor(const std::array<double, 3> &color) {
    backgroundColor = color;
    if (renderer) {
      renderer->SetBackground(backgroundColor.data());
    }
  }

  /// Set the render window size in pixels.
  void setWindowSize(const std::array<int, 2> &size) {
    windowSize = size;
    if (renderWindow) {
      renderWindow->SetSize(windowSize[0], windowSize[1]);
    }
  }

  /// Choose whether to render surfaces, interfaces, or volume cells.
  void setRenderMode(RenderMode mode) { renderMode = mode; }

  /// Convenience setter for the active camera position.
  void setCameraPosition(const std::array<double, 3> &position) {
    if (renderer) {
      renderer->GetActiveCamera()->SetPosition(position.data());
    }
  }

  /// Convenience setter for the active camera view-up vector.
  void setCameraViewUp(const std::array<double, 3> &viewUp) {
    if (renderer) {
      renderer->GetActiveCamera()->SetViewUp(viewUp.data());
    }
  }

  /// Convenience setter for the active camera focal point.
  void setCameraFocalPoint(const std::array<double, 3> &focalPoint) {
    if (renderer) {
      renderer->GetActiveCamera()->SetFocalPoint(focalPoint.data());
    }
  }

  /// Snap camera to axis-aligned views (0=x, 1=y, 2=z).
  void setCameraView(int axis) {
    if (renderer) {
      vtkCamera *camera = renderer->GetActiveCamera();
      switch (axis) {
      case 0: // X
        camera->SetPosition(1, 0, 0);
        camera->SetFocalPoint(0, 0, 0);
        camera->SetViewUp(0, 0, 1);
        break;
      case 1: // Y
        camera->SetPosition(0, 1, 0);
        camera->SetFocalPoint(0, 0, 0);
        camera->SetViewUp(0, 0, 1);
        break;
      case 2: // Z
        camera->SetPosition(0, 0, 1);
        camera->SetFocalPoint(0, 0, 0);
        camera->SetViewUp(0, 1, 0);
        break;
      default:
        VIENNACORE_LOG_WARNING("Invalid axis for camera view.");
      }
    }
  }

  /// Log current camera position, focal point, and view-up vector.
  void printCameraInfo() {
    if (renderer) {
      vtkCamera *camera = renderer->GetActiveCamera();
      double position[3];
      camera->GetPosition(position);
      double focalPoint[3];
      camera->GetFocalPoint(focalPoint);
      double viewUp[3];
      camera->GetViewUp(viewUp);

      VIENNACORE_LOG_INFO("Camera Position: (" + std::to_string(position[0]) +
                          ", " + std::to_string(position[1]) + ", " +
                          std::to_string(position[2]) + ")");
      VIENNACORE_LOG_INFO("Camera Focal Point: (" +
                          std::to_string(focalPoint[0]) + ", " +
                          std::to_string(focalPoint[1]) + ", " +
                          std::to_string(focalPoint[2]) + ")");
      VIENNACORE_LOG_INFO("Camera View Up: (" + std::to_string(viewUp[0]) +
                          ", " + std::to_string(viewUp[1]) + ", " +
                          std::to_string(viewUp[2]) + ")");
    }
  }

  /// Capture the current framebuffer to a PNG file.
  void saveScreenshot(const std::string &fileName, int scale = 1) {
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
        vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(scale); // image quality
    windowToImageFilter->Update();

    VIENNACORE_LOG_INFO("Saving screenshot '" + fileName + "'");

    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(fileName.c_str());
    writer->SetInputData(windowToImageFilter->GetOutput());
    writer->Write();
  }

  /// Render and enter the VTK event loop.
  void render() {
    updateDisplay();
    interactor->Start();
    finalize();
  }

  // Update the display without starting a new event loop (for use during
  // interaction)
  /// Refresh the scene; optionally reuse cached meshes.
  void updateDisplay(bool useCache = true) {
    updateRenderer(useCache);
    renderWindow->Render();
  }

  /// Toggle the on-screen keyboard instruction overlay.
  void toggleInstructionText() {
    if (instructionsAdded) {
      instructionsActor->SetInput("");
      instructionsAdded = false;
    } else {
      instructionsActor->SetInput(instructionsText.c_str());
      instructionsAdded = true;
    }
    renderWindow->Render();
  }

  /// Toggle the material lookup-table legend.
  void toggleScalarBar() {
    if (showScalarBar) {
      renderer->RemoveActor(scalarBar);
      showScalarBar = false;
    } else {
      renderer->AddActor(scalarBar);
      showScalarBar = true;
    }
    renderWindow->Render();
  }

private:
  // forward declaration, requires full definition of Interactor styles
  void initialize();

  void finalize() {
    // Save window size and position for next initialization
    auto size = renderWindow->GetSize();
    windowSize = {size[0], size[1]};

    auto position = renderWindow->GetPosition();
    windowPosition = {position[0], position[1]};
  }

  void updateRenderer(bool useCache = true) {
    if (domains.empty()) {
      VIENNACORE_LOG_WARNING("No domains set for rendering.");
      return;
    }

    // Clear previous actors
    renderer->RemoveAllViewProps();
    renderer->AddActor(instructionsActor);

    // Get min and max material IDs present in the domains
    std::set<Material> uniqueMaterialIds;
    for (const auto &domain : domains) {
      if (domain->getLevelSets().empty()) {
        VIENNACORE_LOG_WARNING("Domain has no level sets for rendering.");
        continue;
      }
      auto materialsInDomain = domain->getMaterialsInDomain();
      uniqueMaterialIds.insert(materialsInDomain.begin(),
                               materialsInDomain.end());
    }

    // Build annotated LUT for material IDs (categorical)
    lut->SetNumberOfTableValues(static_cast<int>(uniqueMaterialIds.size()));
    lut->IndexedLookupOn();
    lut->ResetAnnotations();
    materialMinId = std::numeric_limits<int>::max();
    materialMaxId = std::numeric_limits<int>::min();
    int index = 0;
    for (const auto &materialId : uniqueMaterialIds) {
      auto colorHex = color(materialId);
      auto [r, g, b] = util::hexToRGBArray(colorHex);
      lut->SetTableValue(index, r, g, b, 1.0);
      auto label = to_string_view(materialId);
      int id = static_cast<int>(materialId);
      lut->SetAnnotation(id, label.data());
      ++index;

      materialMinId = std::min(materialMinId, id);
      materialMaxId = std::max(materialMaxId, id);
    }
    lut->Build();
    scalarBar->SetLookupTable(lut);
    scalarBar->SetMaximumNumberOfColors(
        static_cast<int>(uniqueMaterialIds.size()));
    // When using annotations (categorical), don't draw numeric labels
    scalarBar->SetNumberOfLabels(0);
    scalarBar->SetTitle("Materials");

    cachedSurfaceMesh.resize(domains.size());
    cachedInterfaceMesh.resize(domains.size());
    cachedVolumeMesh.resize(domains.size());

    for (std::size_t i{0}; i < domains.size(); ++i) {
      switch (renderMode) {
      case RenderMode::SURFACE: {
        SmartPointer<viennals::Mesh<T>> surfaceMesh;
        if (useCache && cachedSurfaceMesh[i]) {
          surfaceMesh = cachedSurfaceMesh[i];
        } else {
          surfaceMesh = domains[i]->getSurfaceMesh();
          cachedSurfaceMesh[i] = surfaceMesh;
        }
        updatePolyData(surfaceMesh, domainOffsets[i]);
        break;
      }
      case RenderMode::INTERFACE: {
        SmartPointer<viennals::Mesh<T>> interfaceMesh;
        if (useCache && cachedInterfaceMesh[i]) {
          interfaceMesh = cachedInterfaceMesh[i];
        } else {
          interfaceMesh = domains[i]->getSurfaceMesh(true);
          cachedInterfaceMesh[i] = interfaceMesh;
        }
        updatePolyData(interfaceMesh, domainOffsets[i]);
        break;
      }
      case RenderMode::VOLUME: {
        if (useCache && cachedVolumeMesh[i]) {
          updateVolumeMesh(cachedVolumeMesh[i], domainOffsets[i]);
          break;
        }
        viennals::WriteVisualizationMesh<T, D> writer;
        auto const &levelSets = domains[i]->getLevelSets();
        for (auto ls : levelSets) {
          writer.insertNextLevelSet(ls);
        }
        writer.setMaterialMap(domains[i]->getMaterialMap()->getMaterialMap());
        writer.setWriteToFile(false);
        writer.apply();

        auto volumeMesh = writer.getVolumeMesh();
        cachedVolumeMesh[i] = volumeMesh;
        updateVolumeMesh(volumeMesh, domainOffsets[i]);
        break;
      }
      default:
        assert(false && "Unknown render mode.");
      }
    }

    renderer->ResetCamera();
  }

private:
  void updatePolyData(SmartPointer<viennals::Mesh<T>> mesh,
                      const std::array<double, 3> &offset) {
    if (mesh == nullptr) {
      assert(false && "Mesh is null.");
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
    if (useMaterialIds) {
      vtkSmartPointer<vtkIntArray> matIdArray =
          vtkSmartPointer<vtkIntArray>::New();
      matIdArray->SetName("MaterialIds");
      for (const auto &id : *materialIds) {
        matIdArray->InsertNextValue(static_cast<int>(id));
      }
      polyData->GetCellData()->AddArray(matIdArray);
      polyData->GetCellData()->SetActiveScalars("MaterialIds");
      VIENNACORE_LOG_DEBUG("Added MaterialIds array to cell data.");
    }

    vtkSmartPointer<vtkTransform> transform =
        vtkSmartPointer<vtkTransform>::New();
    transform->Translate(offset.data());
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
        vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter->SetTransform(transform);
    transformFilter->SetInputData(polyData);
    transformFilter->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(transformFilter->GetOutput());

    if (useMaterialIds) {
      mapper->SetScalarModeToUseCellData();
      mapper->ScalarVisibilityOn();
      mapper->SelectColorArray("MaterialIds");
      mapper->SetLookupTable(lut);
      mapper->SetScalarRange(materialMinId, materialMaxId);

      if (showScalarBar)
        renderer->AddActor(scalarBar);
    }

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetLineWidth(3.0); // Thicker lines

    renderer->AddActor(actor);
  }

  void updateVolumeMesh(vtkSmartPointer<vtkUnstructuredGrid> volumeMesh,
                        const std::array<double, 3> &offset) {
    if (volumeMesh == nullptr) {
      assert(false && "Volume mesh is null.");
      return;
    }

    vtkSmartPointer<vtkTransform> transform =
        vtkSmartPointer<vtkTransform>::New();
    transform->Translate(offset.data());
    vtkSmartPointer<vtkTransformFilter> transformFilter =
        vtkSmartPointer<vtkTransformFilter>::New();
    transformFilter->SetTransform(transform);
    transformFilter->SetInputData(volumeMesh);
    transformFilter->Update();

    auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(transformFilter->GetOutput());
    mapper->SetScalarModeToUseCellData();
    mapper->ScalarVisibilityOn();
    mapper->SelectColorArray("Material");

    mapper->SetLookupTable(lut);
    mapper->SetScalarRange(materialMinId, materialMaxId);

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    renderer->AddActor(actor);

    // Add scalar bar
    if (showScalarBar)
      renderer->AddActor(scalarBar);
  }

private:
  std::vector<SmartPointer<Domain<T, D>>> domains;
  std::vector<std::array<double, 3>> domainOffsets;

  vtkSmartPointer<vtkRenderer> renderer;
  vtkSmartPointer<vtkRenderWindow> renderWindow;
  vtkSmartPointer<vtkRenderWindowInteractor> interactor;
  vtkSmartPointer<vtkLookupTable> lut;
  vtkSmartPointer<vtkScalarBarActor> scalarBar;

  // text
  const std::string instructionsText =
      "Press 1: Surface | 2: Interface | 3: Volume | "
      "x/y/z: View | s: Screenshot | h: Show/Hide Instructions | b: Show/Hide "
      "Scalar Bar | q/e: Quit";
  vtkSmartPointer<vtkTextActor> instructionsActor;
  bool instructionsAdded = true;

  // static settings
  static std::array<double, 3> backgroundColor;
  static std::array<int, 2> windowSize;
  static std::array<int, 2> windowPosition;
  static RenderMode renderMode;

  // cached meshes
  std::vector<SmartPointer<viennals::Mesh<T>>> cachedSurfaceMesh;
  std::vector<SmartPointer<viennals::Mesh<T>>> cachedInterfaceMesh;
  std::vector<vtkSmartPointer<vtkUnstructuredGrid>> cachedVolumeMesh;
  int materialMinId = -1;
  int materialMaxId = -1;
  bool showScalarBar = true;
};

// Static member definitions
template <typename T, int D>
std::array<double, 3> VTKRenderWindow<T, D>::backgroundColor = {
    84.0 / 255, 89.0 / 255, 109.0 / 255}; // ParaView default background
template <typename T, int D>
std::array<int, 2> VTKRenderWindow<T, D>::windowSize = {900, 700};
template <typename T, int D>
std::array<int, 2> VTKRenderWindow<T, D>::windowPosition = {100, 100};
template <typename T, int D>
RenderMode VTKRenderWindow<T, D>::renderMode = RenderMode::INTERFACE;

namespace impl {
template <typename T, int D>
void InteractorOnChar(vtkRenderWindowInteractor *rwi,
                      VTKRenderWindow<T, D> *window) {

  if (!window || !rwi || rwi->GetDone())
    return;

  auto renderer = rwi->GetRenderWindow()->GetRenderers()->GetFirstRenderer();

  switch (rwi->GetKeyCode()) {
  case '1':
    window->setRenderMode(RenderMode::SURFACE);
    window->updateDisplay(true);
    return;
  case '2':
    window->setRenderMode(RenderMode::INTERFACE);
    window->updateDisplay(true);
    return;
  case '3':
    window->setRenderMode(RenderMode::VOLUME);
    window->updateDisplay(true);
    return;
  case 'e':
    [[fallthrough]];
  case 'q':
    rwi->TerminateApp();
    return;
  case 'x':
    window->setCameraView(0);
    renderer->ResetCamera();
    rwi->GetRenderWindow()->Render();
    return;
  case 'y':
    window->setCameraView(1);
    renderer->ResetCamera();
    rwi->GetRenderWindow()->Render();
    return;
  case 'z':
    window->setCameraView(2);
    renderer->ResetCamera();
    rwi->GetRenderWindow()->Render();
    return;
  case 's': {
    auto time_t =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char timeStamp[20];
    std::strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S",
                  std::localtime(&time_t));
    const std::string fileName =
        "screenshot_" + std::string(timeStamp) + ".png";
    window->saveScreenshot(fileName);
    return;
  }
  case 'h':
    window->toggleInstructionText();
    return;
  case 'b':
    window->toggleScalarBar();
    return;
  case 'p':
    window->printCameraInfo();
    return;
  default:
    return;
  }
}
} // namespace impl
} // namespace viennaps

// Full definition of Custom3DInteractorStyle after VTKRenderWindow is complete
class Custom3DInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
  static Custom3DInteractorStyle *New() {
    VTK_STANDARD_NEW_BODY(Custom3DInteractorStyle);
  }
  vtkTypeMacro(Custom3DInteractorStyle, vtkInteractorStyleTrackballCamera);

  void OnChar() override {
    viennaps::impl::InteractorOnChar(this->Interactor, Window);
  }

  void setRenderWindow(viennaps::VTKRenderWindow<double, 3> *window) {
    Window = window;
  }

private:
  viennaps::VTKRenderWindow<double, 3> *Window = nullptr;
};

// vtkStandardNewMacro(Custom3DInteractorStyle);

class Custom2DInteractorStyle : public vtkInteractorStyleImage {
public:
  static Custom2DInteractorStyle *New() {
    VTK_STANDARD_NEW_BODY(Custom2DInteractorStyle);
  }
  vtkTypeMacro(Custom2DInteractorStyle, vtkInteractorStyleImage);

  void OnLeftButtonDown() override { this->StartPan(); }

  void OnLeftButtonUp() override { this->EndPan(); }

  void OnChar() override {
    viennaps::impl::InteractorOnChar(this->Interactor, Window);
  }

  void setRenderWindow(viennaps::VTKRenderWindow<double, 2> *window) {
    Window = window;
  }

private:
  viennaps::VTKRenderWindow<double, 2> *Window = nullptr;
};

// vtkStandardNewMacro(Custom2DInteractorStyle);

// VTKRenderWindow::initialize() implementation - defined after
// Custom3DInteractorStyle
namespace viennaps {

template <typename T, int D> void VTKRenderWindow<T, D>::initialize() {
  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->SetBackground(backgroundColor.data());

  // add text actor for instructions
  instructionsActor = vtkSmartPointer<vtkTextActor>::New();
  instructionsActor->GetTextProperty()->SetFontSize(14);
  instructionsActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
  instructionsActor->SetPosition(10, 10);
  if (instructionsAdded) {
    instructionsActor->SetInput(instructionsText.c_str());
  } else {
    instructionsActor->SetInput("");
  }

  renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->SetWindowName("ViennaPS");
  renderWindow->SetSize(windowSize.data());
  renderWindow->SetPosition(windowPosition.data());
  renderWindow->AddRenderer(renderer);

  // Initialize interactor
  interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  if constexpr (std::is_same_v<T, double>) {
    if constexpr (D == 2) {
      vtkCamera *cam = renderer->GetActiveCamera();
      cam->ParallelProjectionOn();
      auto style = vtkSmartPointer<Custom2DInteractorStyle>::New();
      style->setRenderWindow(this);
      interactor->SetInteractorStyle(style);
    } else {
      setCameraView(1); // Default view along Y axis
      auto style = vtkSmartPointer<Custom3DInteractorStyle>::New();
      style->setRenderWindow(this);
      interactor->SetInteractorStyle(style);
    }
  } else {
    VIENNACORE_LOG_WARNING("VTKRenderWindow with float precision does not have "
                           "custom interactor.");
  }

  interactor->SetRenderWindow(renderWindow);
  renderWindow->SetInteractor(interactor);

  if (auto style = interactor->GetInteractorStyle()) {
    style->SetDefaultRenderer(renderer);
    style->SetCurrentRenderer(renderer);
  }

  interactor->Initialize();

  lut = vtkSmartPointer<vtkLookupTable>::New();

  // lut->SetNumberOfTableValues(256);
  // lut->SetHueRange(0.667, 0.0); // blue → red
  // lut->SetSaturationRange(1.0, 1.0);
  // lut->SetValueRange(.5, 1.0);
  // lut->Build();

  scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
  // Make scalar bar smaller and labels larger by default
  scalarBar->SetOrientationToVertical();
  // Position in normalized viewport coordinates (x,y)
  scalarBar->SetPosition(0.88, 0.15);
  // Size in normalized viewport coordinates (width,height)
  scalarBar->SetPosition2(0.08, 0.70);
  // Make the color swatch thinner relative to the actor box
  scalarBar->SetBarRatio(0.20);
  // Increase annotation (label) and title font sizes
  scalarBar->GetLabelTextProperty()->SetFontSize(18);
  scalarBar->GetTitleTextProperty()->SetFontSize(20);
  scalarBar->GetLabelTextProperty()->BoldOn();
  scalarBar->GetTitleTextProperty()->BoldOn();
}
} // namespace viennaps

#endif // VIENNALS_VTK_RENDERING