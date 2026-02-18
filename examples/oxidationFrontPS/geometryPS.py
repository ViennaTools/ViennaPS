import viennals as vls

def get_geometry_domains(params):
    """
    Generates the LevelSets and MaterialMap using the provided parameters.
    Matches the interface of geometry.py from ViennaCS examples.
    
    Returns: (list_of_levelsets, material_map, bounds, boundary_conditions, grid_delta)
    """
    print("--- Generating Oxidation Geometry (ViennaPS/LS) ---")

    # Unpack parameters
    grid_delta = params["gridDelta"]
    x_extent = params["xExtent"]
    substrate_height = params["substrateHeight"]
    mask_height = params["maskHeight"]
    hole_radius = params["holeRadius"]
    dimension = int(params.get("dimensions", params.get("dimension", 3)))

    # Ensure dimension is set
    vls.setDimension(dimension)

    # Define Bounds
    # Domain height covers substrate + mask + buffer
    y_max = substrate_height + mask_height + grid_delta

    if dimension == 2:
        bounds = [-x_extent/2.0, x_extent/2.0, 0.0, y_max]
        bc = [
            vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
            vls.BoundaryConditionEnum.INFINITE_BOUNDARY
        ]
    else:  # dimension == 3
        y_extent = params.get("yExtent", x_extent)
        bounds = [-x_extent/2.0, x_extent/2.0, -y_extent/2.0, y_extent/2.0, 0.0, y_max]
        bc = [
            vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
            vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
            vls.BoundaryConditionEnum.INFINITE_BOUNDARY
        ]

    origin = [0.0] * dimension
    normal = [0.0] * dimension
    normal[-1] = 1.0  # Normal points in Z direction

    # --- Layer 0: Bottom Plane (y=0) ---
    bottom = vls.Domain(bounds, bc, grid_delta)
    vls.MakeGeometry(bottom, vls.Plane(origin=origin, normal=normal)).apply()

    # --- Layer 1: Substrate Surface (y=substrateHeight) ---
    origin[-1] = substrate_height
    substrate = vls.Domain(bounds, bc, grid_delta)

    if dimension == 2:
        vls.MakeGeometry(substrate, vls.Plane(origin=origin, normal=normal)).apply()
    else:  # dimension == 3
        # For 3D, we use a cylinder for the substrate to match C++ example
        # Reset origin for cylinder base
        cyl_origin = [0.0] * dimension
        radius = x_extent / 2.0
        vls.MakeGeometry(substrate, vls.Cylinder(origin=cyl_origin, axisDirection=normal,
                                                   height=substrate_height, radius=radius)).apply()

    # Union Bottom and Substrate
    vls.BooleanOperation(substrate, bottom, vls.BooleanOperationEnum.UNION).apply()

    # --- Mask (if maskHeight > 0) ---
    mask = vls.Domain(bounds, bc, grid_delta)
    if mask_height > 0:
        if dimension == 2:
            origin[-1] = substrate_height + mask_height
            vls.MakeGeometry(mask, vls.Plane(origin=origin, normal=normal)).apply()

            # Create hole in mask using Box
            hole = vls.Domain(bounds, bc, grid_delta)
            min_p = [-hole_radius, substrate_height - grid_delta]
            max_p = [hole_radius, substrate_height + mask_height + grid_delta]
            vls.MakeGeometry(hole, vls.Box(minPoint=min_p, maxPoint=max_p)).apply()
        else:  # dimension == 3
            origin[-1] = substrate_height
            radius = x_extent / 2.0 # Mask covers full extent
            vls.MakeGeometry(mask, vls.Cylinder(origin=origin, axisDirection=normal,
                                                  height=mask_height, radius=radius)).apply()

            # Create hole in mask using smaller Cylinder
            hole = vls.Domain(bounds, bc, grid_delta)
            vls.MakeGeometry(hole, vls.Cylinder(origin=origin, axisDirection=normal,
                                                  height=mask_height + 2*grid_delta,
                                                  radius=hole_radius)).apply()

        # Cut hole from mask
        vls.BooleanOperation(mask, hole, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT).apply()

        # Union mask with substrate
        vls.BooleanOperation(mask, level_sets[-1], vls.BooleanOperationEnum.UNION).apply()

    # --- Prepare Output ---
    ls_list = [bottom, substrate, mask]
    
    mat_map = vls.MaterialMap()
    mat_map.insertNextMaterial(0) # Substrate
    mat_map.insertNextMaterial(0) # Substrate (redundant layer)
    mat_map.insertNextMaterial(2) # Mask (Material ID 2)

    print("Geometry generated.")
    return ls_list, mat_map, bounds, bc, grid_delta

if __name__ == "__main__":
    import sys

    # Test with default values if run directly
    test_params = {
        "gridDelta": 0.85,
        "xExtent": 85.0,
        "yExtent": 85.0,
        "substrateHeight": 50.0,
        "maskHeight": 10.0,
        "holeRadius": 20.0,
        "ambientHeight": 8.0,
        "dimensions": 3
    }

    # Allow dimension to be specified on command line
    dimension = test_params["dimensions"]

    print(f"Testing {dimension}D geometry...")

    # Import appropriate viennacs module
    if dimension == 2:
        vls.setDimension(dimension)
        import viennacs2d as vcs
    else:
        vls.setDimension(dimension)
        import viennacs3d as vcs

    # Test 1: get_geometry_domains + manual cell set creation
    print("\nTest 1: Testing get_geometry_domains() + manual fromLevelSets...")
    lss, mat_map, _, _, _ = get_geometry_domains(test_params)

    # Also test creating a cell set from it
    cell_set1 = vcs.DenseCellSet()
    depth1 = test_params["substrateHeight"] + test_params["ambientHeight"]
    cell_set1.setCellSetPosition(True)
    print(f"  Level sets: {len(lss)}, MaterialMap layers: {mat_map.getNumberOfLayers()}")
    print(f"  Depth: {depth1}")
    cell_set1.fromLevelSets(lss, mat_map, depth1)
    cell_set1.buildNeighborhood()
    cell_set1.writeVTU(f"check_cellset_manual_{dimension}d.vtu")
    print(f"  Written check_cellset_manual_{dimension}d.vtu with {cell_set1.getNumberOfCells()} cells")

    # Also write surface mesh
    mesh = vls.Mesh()
    vls.ToSurfaceMesh(lss[-1], mesh).apply()
    vls.VTKWriter(mesh, f"check_geometry_{dimension}d.vtp").apply()
    print(f"  Written check_geometry_{dimension}d.vtp surface mesh")

    # Test 2: make_cell_set (matches C++ interface)
    print("\nTest 2: Testing make_cell_set()...")
    cell_set2 = vcs.DenseCellSet()
    make_cell_set(cell_set2, test_params,
                  substrate_material=0,
                  mask_material=2,
                  ambient_material=3)
    cell_set2.buildNeighborhood()
    cell_set2.writeVTU(f"check_cellset_{dimension}d.vtu")
    print(f"  Written check_cellset_{dimension}d.vtu with {cell_set2.getNumberOfCells()} cells")
    print("\nAll tests complete!")
