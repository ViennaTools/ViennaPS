# ViennaPS Container
#
# ViennaPS is a library for topography simulation in microelectronic
# fabrication processes: etching, deposition, and related steps.
#
# Build:
#   docker build -t viennaps -f docker/Dockerfile .
#
# Run interactively:
#   docker run -it viennaps
#
# Run with mounted workspace:
#   docker run -it -v $(pwd):/workspace viennaps
#
# See: https://github.com/ViennaTools/ViennaPS

FROM ghcr.io/viennatools/vienna-builder:suite-python

LABEL org.opencontainers.image.title="ViennaPS"
LABEL org.opencontainers.image.description="Process simulation for semiconductor fabrication"
LABEL org.opencontainers.image.source="https://github.com/ViennaTools/ViennaPS"

WORKDIR /workspace

# Install ViennaPS from PyPI
# Alpine Linux requires --break-system-packages for global pip installs
RUN pip install ViennaPS --break-system-packages

# Copy test script
COPY docker/test_viennaps.py /workspace/test_viennaps.py

# Verify installation
RUN python -c "import viennaps as vps; print('ViennaPS installed successfully')"

CMD ["python"]
