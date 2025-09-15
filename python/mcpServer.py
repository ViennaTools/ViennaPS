# mcpServer.py
import json
from fastmcp import FastMCP
from fastmcp.resources import FileResource
from fastmcp.server.dependencies import get_context
from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator
import os, pathlib
import vtk  # pip install vtk
import viennaps as ps

mcp = FastMCP("viennaps")
SESSION = {}  # {session_id: {name: Domain}}
RESOURCES = {}  # {uri: {"path": str, "mime": str, "name": str, "desc": str}}


def _abs_out(path: str) -> str:
    p = pathlib.Path(os.path.expanduser(path)).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def store():
    sid = get_context().session_id
    return SESSION.setdefault(sid, {})


def register_png(path: str) -> str:
    p = pathlib.Path(path).resolve()
    if p.exists():
        uri = f"file://{p.as_posix()}"  # use file:// scheme
        res = FileResource(
            uri=uri,
            path=p,
            name=f"Preview {p.name}",
            description="ViennaPS render",
            mime_type="image/png",
        )
        mcp.add_resource(res)  # notifies clients
        return uri
    else:
        raise FileNotFoundError(f"File not found: {p}")


@mcp.tool
def create_domain(gridDelta: float, xExtent: float, yExtent: float = 0.0, dim: int = 2):
    ps.Logger.setLogLevel(ps.LogLevel.ERROR)
    if dim not in (2, 3):
        return {"ok": False, "error": "dim must be 2 or 3"}
    if dim == 3 and yExtent <= 0.0:
        return {"ok": False, "error": "yExtent must be > 0 for 3D domain"}
    if dim == 2:
        d = ps.d2.Domain(gridDelta=gridDelta, xExtent=xExtent)
    else:
        d = ps.d3.Domain(gridDelta=gridDelta, xExtent=xExtent, yExtent=yExtent)
    store()["domain" + str(dim)] = d
    return {"ok": True}


# "_volume.vtu" is appended automatically to the filename
@mcp.tool
def save_volume_mesh(path: str = "~/ViennaPS-Outputs/mesh", dim: int = 2):
    d = store()["domain" + str(dim)]
    full = _abs_out(path)
    try:
        d.saveVolumeMesh(filename=full)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": full + "_volume.vtu"}


@mcp.tool
def save_surface_mesh(path: str = "~/ViennaPS-Outputs/surface.vtp", dim: int = 2):
    d = store()["domain" + str(dim)]
    full = _abs_out(path)
    try:
        d.saveSurfaceMesh(filename=full)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": full}


@mcp.tool
def render_vtu_png(
    vtu_path: str,
    png_path: str,
    width: int = 1200,
    height: int = 900,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    zoom: float = 1.0,
    scalar_array: str | None = None,
):
    """Load VTU with VTK, render offscreen, save PNG, register as resource."""
    vtu_path = _abs_out(vtu_path)
    png_path = _abs_out(png_path)

    # Read VTU
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    data = reader.GetOutput()

    # Print available arrays for debugging
    point_data = data.GetPointData()
    cell_data = data.GetCellData()

    # Surface for fast rendering
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputData(data)
    surf.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surf.GetOutputPort())

    if scalar_array:
        # Check if the array exists in point data or cell data
        if point_data.GetArray(scalar_array):
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray(scalar_array)
        elif cell_data.GetArray(scalar_array):
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray(scalar_array)
        else:
            mapper.ScalarVisibilityOff()

        if point_data.GetArray(scalar_array) or cell_data.GetArray(scalar_array):
            # Set up color mapping
            mapper.SetColorModeToMapScalars()
            mapper.ScalarVisibilityOn()

            # Create a color lookup table for materials
            lut = vtk.vtkLookupTable()
            if scalar_array == "Material":
                # Get the range of material values
                if point_data.GetArray(scalar_array):
                    array_range = point_data.GetArray(scalar_array).GetRange()
                else:
                    array_range = cell_data.GetArray(scalar_array).GetRange()

                # print(f"Material range: {array_range}")

                # Set up the lookup table for discrete materials
                num_materials = int(array_range[1] - array_range[0]) + 1
                lut.SetNumberOfTableValues(num_materials)
                lut.SetRange(array_range)

                # Define colors for different materials
                colors = [
                    [0.8, 0.8, 0.8],  # Light gray
                    [0.2, 0.4, 0.8],  # Blue for first material
                    [0.8, 0.2, 0.2],  # Red for second material
                    [0.2, 0.8, 0.2],  # Green for third material
                    [0.8, 0.8, 0.2],  # Yellow for fourth material
                    [0.8, 0.2, 0.8],  # Magenta for fifth material
                    [0.2, 0.8, 0.8],  # Cyan for sixth material
                ]

                for i in range(num_materials):
                    color = colors[i % len(colors)]
                    lut.SetTableValue(i, color[0], color[1], color[2], 1.0)

                lut.Build()
                mapper.SetLookupTable(lut)
                mapper.SetUseLookupTableScalarRange(True)
    else:
        mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.2, 0.2, 0.2)  # Darker background for better contrast

    # Add a color bar if we're showing scalar data
    # if scalar_array and (point_data.GetArray(scalar_array) or cell_data.GetArray(scalar_array)):
    #     scalar_bar = vtk.vtkScalarBarActor()
    #     scalar_bar.SetLookupTable(mapper.GetLookupTable())
    #     scalar_bar.SetTitle(scalar_array)
    #     scalar_bar.SetNumberOfLabels(5)
    #     scalar_bar.SetPosition(0.85, 0.1)
    #     scalar_bar.SetWidth(0.1)
    #     scalar_bar.SetHeight(0.8)
    #     ren.AddActor2D(scalar_bar)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(width, height)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)

    # Camera setup
    cam = ren.GetActiveCamera()
    ren.ResetCamera()
    cam.Azimuth(azimuth)
    cam.Elevation(elevation)
    cam.Zoom(zoom)

    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(png_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    uri = register_png(png_path)
    return {"png_path": png_path, "resource_uri": uri}


# Geometry tools


# --- params with robust material parsing ---
def _coerce_material(v):
    if isinstance(v, ps.Material):
        return v
    if isinstance(v, int):
        return ps.Material(v)
    raise TypeError("material must be int|Material")


class PlaneParams(BaseModel):
    position: float
    material: ps.Material = ps.Material.Si

    @field_validator("material", mode="before")
    def _mat(cls, v):
        return _coerce_material(v)


class TrenchParams(BaseModel):
    width: float
    depth: float
    taper_angle: float = 0.0
    material: ps.Material = ps.Material.Si
    make_mask: bool = False

    @field_validator("material", mode="before")
    def _mat(cls, v):
        return _coerce_material(v)


class HoleParams(BaseModel):
    radius: float
    depth: float
    taper_angle: float = 0.0
    material: ps.Material = ps.Material.Si
    make_mask: bool = False

    @field_validator("material", mode="before")
    def _mat(cls, v):
        return _coerce_material(v)


class MakePlane(BaseModel):
    geometry_type: Literal["plane"]
    params: PlaneParams


class MakeTrench(BaseModel):
    geometry_type: Literal["trench"]
    params: TrenchParams


class MakeHole(BaseModel):
    geometry_type: Literal["hole"]
    params: HoleParams


GeometrySpec = Annotated[
    Union[MakePlane, MakeTrench, MakeHole], Field(discriminator="geometry_type")
]


class MakeGeometryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # accept object OR JSON string
    spec: Union[GeometrySpec, str]
    dim: int = 2  # 2D or 3D

    @field_validator("spec", mode="before")
    def parse_spec(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


@mcp.tool
def create_geometry(payload: MakeGeometryInput):
    d = store()["domain" + str(payload.dim)]
    spec = payload.spec

    if payload.dim == 2:
        psd = ps.d2
    else:
        psd = ps.d3

    if spec.geometry_type == "plane":  # type: ignore
        p = spec.params  # type: ignore
        psd.MakePlane(domain=d, height=p.position, material=p.material).apply()
    elif spec.geometry_type == "trench":  # type: ignore
        p = spec.params  # type: ignore
        if p.make_mask:
            psd.MakeTrench(
                domain=d,
                trenchWidth=p.width,
                trenchDepth=0.0,
                trenchTaperAngle=0.0,
                material=p.material,
                maskHeight=p.depth,
                maskTaperAngle=p.taper_angle,
            ).apply()
        else:
            psd.MakeTrench(
                domain=d,
                trenchWidth=p.width,
                trenchDepth=p.depth,
                trenchTaperAngle=p.taper_angle,
                material=p.material,
            ).apply()
    elif spec.geometry_type == "hole":  # type: ignore
        p = spec.params  # type: ignore
        if p.make_mask:
            psd.MakeHole(
                domain=d,
                holeRadius=p.radius,
                holeDepth=0.0,
                holeTaperAngle=0.0,
                material=p.material,
                maskHeight=p.depth,
                maskTaperAngle=p.taper_angle,
            ).apply()
        else:
            psd.MakeHole(
                domain=d,
                holeRadius=p.radius,
                holeDepth=p.depth,
                holeTaperAngle=p.taper_angle,
                material=p.material,
            ).apply()

    return {"ok": True}


# Process tools


class IsotropicProcessParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    rate: float
    depo_material: ps.Material = ps.Material.Undefined
    mask_material: ps.Material = ps.Material.Undefined

    @field_validator("depo_material", "mask_material", mode="before")
    def _mat(cls, v):
        return _coerce_material(v)


class SingleParticleProcessParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    rate: float
    sticking_probability: float = 1.0
    source_exponent: float = 1.0
    depo_material: ps.Material = ps.Material.Undefined
    mask_material: ps.Material = ps.Material.Undefined

    @field_validator("depo_material", "mask_material", mode="before")
    def _mat(cls, v):
        return _coerce_material(v)


class SingleParticleProcess(BaseModel):
    process_type: Literal["singleParticle"]
    params: SingleParticleProcessParams


class IsotropicProcess(BaseModel):
    process_type: Literal["isotropic"]
    params: IsotropicProcessParams


ProcessSpec = Annotated[
    Union[IsotropicProcess, SingleParticleProcess], Field(discriminator="process_type")
]


class MakeProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # accept object OR JSON string
    spec: Union[ProcessSpec, str]
    duration: float  # in seconds
    dim: int = 2  # 2D or 3D

    @field_validator("spec", mode="before")
    def parse_spec(cls, v):
        if isinstance(v, str):
            return json.load(v)
        return v


@mcp.tool
def run_process(payload: MakeProcessInput):
    d = store()["domain" + str(payload.dim)]
    spec = payload.spec
    duration = payload.duration

    if payload.dim == 2:
        psd = ps.d2
    else:
        psd = ps.d3

    if spec.process_type == "isotropic":  # type: ignore
        p = spec.params  # type: ignore
        proc = psd.IsotropicProcess(
            rate=p.rate,
            maskMaterial=p.mask_material,
        )
        if p.rate > 0 and p.depo_material != ps.Material.Undefined:
            d.duplicateTopLevelSet(p.depo_material)
    elif spec.process_type == "singleParticle":  # type: ignore
        p = spec.params  # type: ignore
        proc = psd.SingleParticleProcess(
            rate=p.rate,
            stickingProbability=p.sticking_probability,
            sourceExponent=p.source_exponent,
            maskMaterial=p.mask_material,
        )
        if p.rate > 0 and p.depo_material != ps.Material.Undefined:
            d.duplicateTopLevelSet(p.depo_material)
    else:
        return {"ok": False, "error": "Unknown process type"}

    psd.Process(d, proc, duration).apply()  # type: ignore

    return {"ok": True}


if __name__ == "__main__":
    mcp.run()  # stdio
