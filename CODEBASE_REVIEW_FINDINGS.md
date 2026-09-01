# ViennaPS Codebase Review Findings

Generated: 2026-09-01

Scope: static review of the repository structure, CMake/CI setup, Python packaging and bindings, core domain/process headers, compact CSV utilities, GDS handling, GPU install/Docker surfaces, examples, and tests. I did not edit source code. An existing `graphify-out/graph.json` was not present, so this review used direct repository inspection instead of querying an existing graph.

Note: the working tree already had local edits in `gpu/Dockerfile`, `pyproject.toml`, and `python/CMakeLists.txt`; I treated those as part of the current tree and did not change them.

## Executive Summary

The highest-impact issues are around persistence and hidden mutation: `.vpsd` round-trips currently drop cell-set contents, `Reader::apply()` clears the destination before it knows the file is valid, and `saveLevelSetMesh()` mutates the domain while appearing to be an output-only operation. The process context also has sticky flags that can leak behavior between reused `Process` instances. On the quality side, the Python tests can report success without running their test functions, and several legacy Python test files use stale import names.

## Findings

### 1. `.vpsd` round-trips drop cell-set data

Severity: High

Evidence:
- `include/viennaps/psWriter.hpp:17-21` documents that `.vpsd` contains level sets, cell data, material mappings, and setup information.
- `include/viennaps/psWriter.hpp:115-126` writes only a `hasCellSet` flag, then warns that cell-set serialization is not implemented.
- `include/viennaps/psReader.hpp:159-169` reads the flag, then warns that cell-set deserialization is not implemented.
- `tests/readAndWrite/readAndWrite.cpp:23-78` compares only level-set count, material map, and grid delta; it explicitly does not compare level-set values or cell-set contents.

Impact: Any domain that has generated volume/cell data can be saved and reloaded with the cell-set silently absent. A user may trust the `.vpsd` file as a complete simulation checkpoint when it is not.

Suggested fix: either implement cell-set serialization/deserialization, or make writer/reader fail loudly when a cell set is present and persistence is incomplete. Add a round-trip test that checks `getCellSet()`, depth, cell data arrays, and representative material assignments.

### 2. `Reader::apply()` destroys the existing domain before validating the input

Severity: High

Evidence:
- `include/viennaps/psReader.hpp:60-68` opens the file and immediately calls `domain->clear()`.
- Header, version, dimension, setup, level-set, and material-map validation happen afterward in `include/viennaps/psReader.hpp:70-157`.
- Stream reads are generally unchecked, for example `include/viennaps/psReader.hpp:71-82`, `include/viennaps/psReader.hpp:102-148`, and `include/viennaps/psDomainSetup.hpp:186-197`.

Impact: Reading a wrong-dimension, truncated, or otherwise invalid file into an already populated domain can erase the caller's current domain and then throw or return. Corrupt files can also propagate partially read values before an error is detected.

Suggested fix: deserialize into a temporary `Domain` and swap/assign only after the full read succeeds. Check stream state after every fixed-size read and after nested deserializers.

### 3. `saveLevelSetMesh()` mutates simulation state through a getter

Severity: High

Evidence:
- `include/viennaps/psDomain.hpp:531-540` implements `getLevelSetMesh()` by calling `viennals::Expand(...).apply()` directly on each stored level set.
- `include/viennaps/psDomain.hpp:607-615` calls `getLevelSetMesh()` from `saveLevelSetMesh()`.
- Python exposes both methods in `python/pyWrapDimension.hpp:577-590`.

Impact: A seemingly read-only mesh export permanently expands the domain's internal level sets. This can change later simulation behavior or mask bugs in tests that save meshes before comparing domains.

Suggested fix: copy each level set before expanding it for visualization/export, or rename/document the method as mutating and provide a non-mutating alternative.

### 4. Process context flags can leak between runs

Severity: High

Evidence:
- The flag fields are stored in `include/viennaps/process/psProcessContext.hpp:51-57`.
- `updateFlags()` sets some values directly but latches others: `flags.useCoverages = flags.isALP || flags.useCoverages` at `include/viennaps/process/psProcessContext.hpp:71`, and `domainHasPeriodicBoundaries` is only ever set to `true` in `include/viennaps/process/psProcessContext.hpp:73-80`.
- `hasSurfaceDiffusion` and `hasSurfaceDesorption` are only updated when a surface model exists in `include/viennaps/process/psProcessContext.hpp:82-88`.
- `Process` can be reused via setters in `include/viennaps/process/psProcess.hpp:72-81`, and examples use this style, for example `examples/cantileverWetEtching/cantileverWetEtching.py:63-67`.

Impact: Reusing a `Process` with a new domain or model can keep stale periodic-boundary, coverage, diffusion, or desorption behavior from the previous run. That can select a different flux engine or execute extra work with the wrong assumptions.

Suggested fix: recompute all flags from scratch at the start of `updateFlags()`. If coverage initialization needs to request coverage use later, keep that as a separate runtime state instead of mixing it into model/domain capability flags.

### 5. Surface diffusion can skip valid targets because of the early-exit scan

Severity: Medium

Evidence:
- `include/viennaps/process/psFluxProcessStrategy.hpp:501-510` stops scanning diffusion coefficients after the first target name that exists, even if that coefficient is `<= 0`.
- The actual diffusion loop in `include/viennaps/process/psFluxProcessStrategy.hpp:521-548` would process later positive coefficients, but the early return can prevent that loop from running.

Impact: A model with multiple diffusion entries can skip all diffusion when an earlier target exists with a disabled coefficient and a later target has a positive coefficient.

Suggested fix: scan until a target with `coefficient > 0` is found, or remove the pre-scan and let the main loop decide whether any work was performed.

### 6. Public `Domain` methods are unsafe on default or empty domains

Severity: Medium

Evidence:
- Python exposes a default domain constructor in `python/pyWrapDimension.hpp:472-475`.
- `include/viennaps/psDomain.hpp:407` returns `levelSets_.back()` without checking for emptiness.
- `include/viennaps/psDomain.hpp:433-439` returns an uninitialized bounding box when no level sets exist.
- `include/viennaps/psDomain.hpp:448-449`, `include/viennaps/psDomain.hpp:453-458`, and `include/viennaps/psDomain.hpp:498-508` can dereference an empty level-set vector or null material map.
- `include/viennaps/psDomain.hpp:141-167` deep-copies `domain->materialMap_` without checking for null.
- `tests/domain/domain.cpp:14-18` checks only that default construction succeeds.

Impact: Straightforward calls like `Domain().getSurface()`, `Domain().getBoundaryConditions()`, `Domain().getMaterialsInDomain()`, `Domain().print()`, or `deepCopy()` from a domain constructed from raw level sets can crash or return undefined data.

Suggested fix: define explicit empty-domain behavior for public APIs. Throw/log clear errors for methods requiring at least one level set and a material map, zero-initialize returned bounding boxes if that is the intended behavior, and add tests for the Python-facing default-domain cases.

### 7. GDS parsing lacks basic failure containment

Severity: Medium

Evidence:
- `include/viennaps/gds/psGDSReader.hpp:43-45` calls `geometry->finalize()` without checking that `geometry` is set.
- `include/viennaps/gds/psGDSReader.hpp:299-335` opens a `FILE*` and returns on `EndLib` without closing it.
- Low-level reads ignore `fread` results in `include/viennaps/gds/psGDSReader.hpp:101-168` and `include/viennaps/gds/psGDSReader.hpp:309-313`.
- `tests/gdsReader/gdsReader.cpp:14-17` constructs a reader but never calls `reader.apply()`.

Impact: Missing geometry, truncated input, or repeated successful reads can lead to null dereferences, undefined parse values, and leaked file handles. The current GDS test does not exercise the parser.

Suggested fix: use RAII file handling, validate `geometry` before parsing/finalizing, check every read count, fail on malformed records, and add fixture-based tests for valid, truncated, missing-reference, and repeated-read cases.

### 8. GDS hierarchy/bounds handling can hang or compute wrong transformed bounds

Severity: Medium

Evidence:
- `include/viennaps/gds/psGDSGeometry.hpp:352-363` only warns about missing structure references.
- `include/viennaps/gds/psGDSGeometry.hpp:378-464` loops until every structure is processed, but there is no progress detection for missing or cyclic references.
- In release builds, the `assert(refStr)` at `include/viennaps/gds/psGDSGeometry.hpp:389-391` will not protect the following dereference.
- Rotation updates reuse already-mutated coordinates in `include/viennaps/gds/psGDSGeometry.hpp:415-422`.

Impact: A GDS hierarchy with missing or cyclic SREFs can hang during bounding-box calculation. Rotated references can get incorrect bounds because `minPoint_y` and `maxPoint_y` are computed from already-updated x values.

Suggested fix: topologically process references with cycle detection, convert missing references into hard parse errors or skipped refs, and compute transformed points from saved original coordinates.

### 9. GDS blur and polygon offset state has surprising behavior

Severity: Medium

Evidence:
- `include/viennaps/gds/psGDSGeometry.hpp:288-308` takes a `gridRefinement` argument that shadows the member field. The computed value is logged and used locally for sigma scaling, but `applyBlur()` later uses the member in `include/viennaps/gds/psGDSGeometry.hpp:143-146`.
- The same `addBlur()` appends to `sigmas` with `push_back` and never clears previous sigmas.
- `include/viennaps/gds/psGDSGeometry.hpp:548-570` accepts polygon offsets but inserts each original point without applying them.

Impact: Caller-specified blur refinement can be ignored during proximity evaluation, repeated `addBlur()` calls can accumulate stale sigma values, and polygon offsets are ineffective for callers that use nonzero offsets.

Suggested fix: assign `this->gridRefinement`, clear/replace `sigmas` when replacing `weights`, and insert offset-adjusted polygon nodes.

### 10. The Python config parser truncates normal values in a common edge case

Severity: Medium

Evidence:
- `python/__init__.py:102` does `line = line[: line.find("#")]`.
- When a non-comment line has no `#`, `find()` returns `-1`; newline-terminated lines happen to lose only the newline, but a final line without a trailing newline loses the last real character.
- `python/__init__.py:105-106` then splits using `line.find("=")` even if `=` is absent.
- Many Python examples depend on this helper, for example `examples/holeEtching/holeEtching.py:18`, `examples/trenchDeposition/trenchDeposition.py:19`, and `examples/DRAMWiggling/DRAMWiggling.py:21`.

Impact: A valid config file whose last line is `key=123` without a final newline is parsed as `12`. Malformed lines can also create accidental keys instead of being rejected.

Suggested fix: use `line.split("#", 1)[0].strip()`, skip empty lines, require `"=" in line`, split once, and catch `ValueError` rather than a bare `except`.

### 11. Python tests can false-pass and are not wired into CI

Severity: Medium

Evidence:
- `python/tests/run_all_tests.py:15-21` imports each test module but does not call its `test_*` functions.
- Those functions only run under `if __name__ == "__main__"` in `python/tests/test_basic_functionality.py:37-40`, `python/tests/test_models.py:46-48`, and `python/tests/test_integration.py:64-68`.
- Several tests catch exceptions and print messages without failing, for example `python/tests/test_models.py:14-20`, `python/tests/test_models.py:35-43`, and `python/tests/test_integration.py:56-61`.
- The Python workflow smoke-tests only `import viennaps`, for example `.github/workflows/python.yml:126-131`, `.github/workflows/python.yml:145-146`, and `.github/workflows/python.yml:161-162`.
- Several Python tests under `tests/` still import legacy names such as `viennaps2d` and `viennaps3d`, for example `tests/CSVFileProcess/test_csv_2d.py:3`, `tests/isotropicProcess/test_isotropic_3d.py:1`, and `tests/singleParticleProcess/test_single_particle_2d.py:1`, while current bindings expose `viennaps.d2` and `viennaps.d3` in `python/__init__.py:51-55`.

Impact: Python binding regressions can slip through CI. The custom runner reports success after imports, and the legacy test files would fail if run against the current package namespace.

Suggested fix: convert Python tests to pytest/unittest, use real assertions, wire them into CI after wheel installation, and update legacy imports to `import viennaps.d2 as vps` / `import viennaps.d3 as vps`.

### 12. Writer suffix and stream handling are fragile

Severity: Low to Medium

Evidence:
- `include/viennaps/psWriter.hpp:59-62` checks suffixes with `fileName.find(".vpsd") != fileName.length() - 5`.
- For filenames shorter than five characters, `fileName.length() - 5` underflows. If the name also has no `.vpsd`, both sides can compare as `npos`, so the suffix is not appended.
- `include/viennaps/psWriter.hpp:65-128` writes without checking that the stream opened or remained good.

Impact: Short output names can produce extensionless files despite the documented `.vpsd` behavior. Unwritable paths can fail silently.

Suggested fix: use an ends-with check and validate `fout.is_open()`/`fout.good()` after opening and after the final write.

### 13. README uses the wrong CMake option for ViennaPS precompiled/shared builds

Severity: Low

Evidence:
- The project option is `VIENNAPS_PRECOMPILE_HEADERS` in `CMakeLists.txt:22`.
- The README instructs users to configure with `-DVIENNALS_PRECOMPILE_HEADERS=ON` in `README.md:148-154`.
- The top-level CMake maps ViennaPS' option to ViennaLS in `CMakeLists.txt:178-184`.

Impact: Users following the shared-library instructions toggle the dependency option rather than the ViennaPS option described by the section, so they may not get the intended ViennaPS precompiled/shared behavior.

Suggested fix: update the README command to `cmake -B build -DVIENNAPS_PRECOMPILE_HEADERS=ON`.

### 14. CMake configure writes a generated header into the source tree

Severity: Low

Evidence:
- `CMakeLists.txt:131-133` runs `configure_file()` from `cmake/psVersion.hpp.in` to `${PROJECT_SOURCE_DIR}/include/viennaps/psVersion.hpp`.

Impact: Out-of-source configures can still mutate the source checkout, dirtying working trees and making read-only source builds harder.

Suggested fix: generate into `${PROJECT_BINARY_DIR}/generated/include/viennaps/psVersion.hpp`, add that directory to the build interface include path, and keep install/export paths consistent.

### 15. Python package metadata does not declare supported Python versions

Severity: Low

Evidence:
- `pyproject.toml:8-15` declares project metadata and dependencies but no `requires-python`.
- CI intentionally skips older Python versions in `.github/workflows/python.yml` while building current wheels.

Impact: Package installers can attempt source builds on unsupported Python versions and fail later during build or import.

Suggested fix: declare the minimum supported Python version in `[project]`, matching the wheel matrix and syntax used by the package/scripts.

## Verification Notes

- Read-only checks performed included repository inventory (`rg --files`), targeted line inspections with `nl`/`sed`, TODO/FIXME scans, secret-pattern scan, current git diff review, and `ctest -N --test-dir build`.
- `ctest -N --test-dir build` listed 41 configured tests but reported missing executables from the existing build directory, so I did not run the test suite or claim it passes.
- I avoided configuring or rebuilding because `CMakeLists.txt` currently writes a generated header into the source tree, and the request explicitly said not to edit code.
