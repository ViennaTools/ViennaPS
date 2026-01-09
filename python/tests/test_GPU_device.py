import viennaps as vps

vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

try:
    context = vps.gpu.Context()
except AttributeError:
    print("ERROR: Python bindings have not been built with GPU support")
    exit()

context = vps.gpu.Context.createContext()  # create with default module path
print("SUCCESS")
context.destroy()
