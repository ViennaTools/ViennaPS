import viennaps3d as vps

vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

try:
    context = vps.gpu.Context()
except AttributeError:
    print("ERROR: Python bindings have not been built with GPU support")
    exit()

context.create()
print("SUCCESS")
context.destroy()
