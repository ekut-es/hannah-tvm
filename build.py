import subprocess
import pathlib
import shutil   
import os

def build():
    print("Put your build code here!")
    
    
    # Check that cmake is installed
    try:
        subprocess.run(["cmake", "--version"])
    except FileNotFoundError:
        raise Exception("CMake is not installed!")
    
    print("CMake is installed!")
    print("Building the project...")
    print("Done!")
    
    # Check that ninja is installed
    try:
        subprocess.run(["ninja", "--version"])
    except FileNotFoundError:
        raise Exception("Ninja is not installed!")
    
    print("Ninja is installed!")
    
    # Create the build directory
    build_dir = pathlib.Path("external/tvm/build").absolute()
    
    print("Creating the build directory...")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    print("Checking if tvm is already built...")
    
    if not (build_dir / "libtvm.so").exists():
        print("TVM is not built!")
        print("Building TVM...")


        if os.getenv("TVM_CONFIG") is None: 
            with (build_dir / "config.cmake").open("w") as f:
                f.write("set(USE_CUDA ON)\n")
                f.write("set(USE_LLVM ON)\n")
                f.write("set(USE_RPC ON)\n")
                f.write("set(USE_SORT ON)\n")
                f.write("set(USE_GRAPH_RUNTIME ON)\n")
                f.write("set(USE_MICRO ON)\n")
                f.write("set(USE_UMA ON)\n")
        else:
            # Copy the config file
            shutil.copy(os.getenv("TVM_CONFIG"), build_dir / "config.cmake")

        subprocess.run(["cmake", "-G", "Ninja", ".."], cwd=build_dir)
        subprocess.run(["ninja"], cwd=build_dir)    
    
    
    
    

if __name__ == "__main__":    
    build()  # This is a dummy value for now