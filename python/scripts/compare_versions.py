import argparse
import os
import subprocess
import shutil
import vtk

# List of examples to compare
EXAMPLES = [
    {
        "name": "Trench_Deposition",
        "dir": "examples/trenchDeposition",
        "script": "trenchDeposition.py",
        "args": ["-D", "2", "config.txt"],
        "output_file": "final_hull.vtp",
    },
    {
        "name": "Hole_Etching",
        "dir": "examples/holeEtching",
        "script": "holeEtching.py",
        "args": ["-D", "2", "config.txt"],
        "output_file": "final_noIntermediate.vtp",
    },
    {
        "name": "Blazed_Gratings_Etching",
        "dir": "examples/blazedGratingsEtching",
        "script": "blazedGratingsEtching.py",
        "args": ["config.txt"],
        "output_file": "BlazedGratingsEtch_P4.vtp",
    },
    {
        "name": "Bosch_simulate",
        "dir": "examples/boschProcess",
        "script": "boschProcessSimulate.py",
        "args": ["-D", "2", "config.txt"],
        "output_file": "boschProcessSimulate_final_volume.vtu",
    },
    {
        "name": "Bosch_emulate",
        "dir": "examples/boschProcess",
        "script": "boschProcessEmulate.py",
        "args": ["-D", "2", "config.txt"],
        "output_file": "boschProcessEmulate_final_volume.vtu",
    },
]


def get_reader(filename):
    """Returns the appropriate VTK reader for the file extension."""
    if filename.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif filename.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError(f"Unsupported file extension: {filename}")
    reader.SetFileName(filename)
    reader.Update()
    return reader


def create_actor(reader, color, scalar_range=None):
    """Creates a VTK actor from a reader."""
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Check for Material array in vtkUnstructuredGrid (VTU)
    output = reader.GetOutput()
    use_material = False
    if output.IsA("vtkUnstructuredGrid"):
        cell_data = output.GetCellData()
        if cell_data.HasArray("Material"):
            use_material = True
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray("Material")
            if scalar_range:
                mapper.SetScalarRange(scalar_range)

    if not use_material:
        mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if not use_material:
        actor.GetProperty().SetColor(color)

    actor.GetProperty().SetEdgeColor(0, 0, 0)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().EdgeVisibilityOn()
    return actor


def render_comparison(file1, file2, title, output_path):
    """Renders two geometry files side-by-side using VTK."""
    print(f"Generating comparison image: {output_path}")

    reader1 = get_reader(file1)
    reader2 = get_reader(file2)

    # Calculate scalar range for materials if applicable
    scalar_range = None
    if reader1.GetOutput().IsA("vtkUnstructuredGrid") and reader2.GetOutput().IsA(
        "vtkUnstructuredGrid"
    ):
        out1 = reader1.GetOutput()
        out2 = reader2.GetOutput()
        if out1.GetCellData().HasArray("Material") and out2.GetCellData().HasArray(
            "Material"
        ):
            r1 = out1.GetCellData().GetArray("Material").GetRange()
            r2 = out2.GetCellData().GetArray("Material").GetRange()
            scalar_range = (min(r1[0], r2[0]), max(r1[1], r2[1]))

    # Left Viewport (Version 1)
    renderer1 = vtk.vtkRenderer()
    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer1.SetBackground(1.0, 1.0, 1.0)  # White background
    actor1 = create_actor(reader1, (0.8, 0.3, 0.3), scalar_range)  # Reddish
    renderer1.AddActor(actor1)

    # Right Viewport (Version 2)
    renderer2 = vtk.vtkRenderer()
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
    renderer2.SetBackground(1.0, 1.0, 1.0)
    actor2 = create_actor(reader2, (0.3, 0.3, 0.8), scalar_range)  # Blueish
    renderer2.AddActor(actor2)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 600)
    render_window.AddRenderer(renderer1)
    render_window.AddRenderer(renderer2)
    render_window.SetOffScreenRendering(1)

    # Reset cameras to fit geometry
    renderer1.ResetCamera()
    renderer2.ResetCamera()

    # Sync cameras
    cam1 = renderer1.GetActiveCamera()
    cam2 = renderer2.GetActiveCamera()
    cam2.SetPosition(cam1.GetPosition())
    cam2.SetFocalPoint(cam1.GetFocalPoint())
    cam2.SetViewUp(cam1.GetViewUp())

    # # Zoom to decrease visual distance (reduce whitespace)
    # cam1.Zoom(1.4)
    # cam2.Zoom(1.4)

    render_window.Render()

    # Save to file
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()


def run_example(base_dir, example, version_label, output_dir):
    """Runs a single example in the given base directory."""
    venv_python = os.path.join(base_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        print(f"Error: Python executable not found at {venv_python}")
        return None

    example_dir = os.path.join(base_dir, example["dir"])
    cmd = [venv_python, example["script"]] + example["args"]

    print(f"Running {example['name']} in {base_dir}...")
    try:
        # Capture output to avoid cluttering stdout
        subprocess.run(
            cmd,
            cwd=example_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {example['name']}: {e.stderr.decode()}")
        return None

    src_file = os.path.join(example_dir, example["output_file"])
    if not os.path.exists(src_file):
        print(f"Expected output file not found: {src_file}")
        return None

    dst_filename = (
        f"{example['name']}_{version_label}.{example['output_file'].split('.')[-1]}"
    )
    dst_file = os.path.join(output_dir, dst_filename)
    shutil.copy(src_file, dst_file)
    return dst_file


def main():
    parser = argparse.ArgumentParser(description="Compare ViennaPS versions")
    parser.add_argument("dir1", help="Path to directory with version 1")
    parser.add_argument("dir2", help="Path to directory with version 2")
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_results",
        help="Output directory for images",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for example in EXAMPLES:
        print(f"\nProcessing {example['name']}...")

        out1 = run_example(os.path.abspath(args.dir1), example, "v1", args.output)
        out2 = run_example(os.path.abspath(args.dir2), example, "v2", args.output)

        if out1 and out2:
            image_path = os.path.join(args.output, f"{example['name']}_comparison.png")
            render_comparison(out1, out2, example["name"], image_path)
        else:
            print(f"Skipping comparison for {example['name']} due to missing outputs.")


if __name__ == "__main__":
    main()
