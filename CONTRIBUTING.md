# Contributing

## Running CI locally

All Ubuntu CI jobs can be reproduced locally using [act](https://github.com/nektos/act) (nektos), which runs GitHub Actions workflows inside Docker containers.

### Prerequisites

**Docker** must be installed and the daemon must be running.

**act** (nektos — not the Java-based package from `apt`):

```bash
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### Fixing format errors

The format check only reports what is wrong — it never modifies files. To auto-apply all formatting corrections, run this once from the `ViennaPS/` root before re-running the check:

```bash
[ ! -d _viennals ] && git clone --depth 1 --branch master https://github.com/ViennaTools/ViennaLS.git _viennals

docker run --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ghcr.io/viennatools/vienna-builder:suite-python \
  sh -c "cmake -B /tmp/fmtbuild -DCPM_ViennaLS_SOURCE=/workspace/_viennals && cmake --build /tmp/fmtbuild --target format"
```

This uses the exact same formatter (container's `clang-format` + `cmake-format` with the ViennaCore config) that the check runs against, so the result is guaranteed to pass.

### Commands

Run all three from the `ViennaPS/` root:

```bash
# Format check
act push -j check-coding-style -W .github/workflows/format.yml

# C++ build + tests (Release)
act push -j test --matrix 'os:ubuntu-latest' --matrix 'config:Release' \
  -W .github/workflows/build.yml

# Python bindings
act push -j test-bindings --matrix 'os:ubuntu-latest' \
  -W .github/workflows/python.yml
```

### First run

Three Docker images are pulled on first use:

| Workflow | Image |
|---|---|
| Format | `ghcr.io/viennatools/vienna-builder:suite-python` |
| Build / Tests | `ghcr.io/viennatools/vienna-builder:suite` |
| Python bindings | `ghcr.io/viennatools/vienna-builder:cuda-suite-python` (~4 GB) |

Subsequent runs reuse the cached images.

### Notes

- **Artifact upload** is skipped locally — the builder images do not include Node.js. This only affects the `📦 Upload Artifact` step and does not indicate a build failure.
- **Do not use `--use-gitignore`** — act will copy committed file versions instead of your working tree, so local edits are ignored.
- **Windows / macOS** jobs cannot be run with `act` and must be tested on real GitHub Actions runners.
