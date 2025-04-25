import csv
import viennals2d as vls
import viennaps2d as vps

def read_path_dual(domain, target, params):
    read_path(domain, params, is_target=False)
    read_path(target, params, is_target=True)

def read_path(domain, params, is_target=False):
    path = params.targetFile if is_target else params.pathFile
    reader = vps.CSVReader(path)
    data = reader.readContent()
    if data is None:
        print(f"[Warning] Failed to read CSV data from {path}")
        return

    mesh = vls.Mesh()
    node_ids = []

    if params.buffer > 0. and data:
        x = data[0][0] - params.gridDelta
        y = data[0][1] - params.offSet
        node_ids.append(mesh.insertNextNode([x, y, 0.]))

    for row in data:
        if len(row) < 2:
            continue
        x = row[0] + params.buffer
        y = row[1] - params.offSet
        node_ids.append(mesh.insertNextNode([x, y, 0.]))

    if params.buffer > 0. and data:
        x = data[-1][0] + 2. * params.buffer + params.gridDelta
        y = data[-1][1] - params.offSet
        node_ids.append(mesh.insertNextNode([x, y, 0.]))

    for i in range(1, len(node_ids)):
        mesh.insertNextLine([node_ids[i], node_ids[i - 1]])

    extent = params.getExtent()
    x_max = extent[0] / 2. if params.halfGeometry else extent[0]
    bounds = [0., x_max, -1., 1.]

    bcs = [
        vls.BoundaryConditionEnum.PERIODIC_BOUNDARY if params.periodicBoundary else vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ]
    cut = vls.Domain(bounds, bcs, params.gridDelta)
    vls.FromSurfaceMesh(cut, mesh).apply()
    domain.applyBooleanOperation(cut, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT)
    