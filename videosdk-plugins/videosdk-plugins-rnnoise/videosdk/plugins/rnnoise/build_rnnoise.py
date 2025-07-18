import os
import platform
import subprocess
import shutil
import tempfile

import sys
from shutil import which

def check_command(cmd, install_tip):
    if not which(cmd):
        raise EnvironmentError(f"{cmd} not found. Install it: {install_tip}")

def run_subprocess(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}: {e.stderr}")
        raise

def build_rnnoise():
    os_sys = platform.system()
    repo_url = "https://github.com/xiph/rnnoise.git"
    
    git_tip = "Install git (e.g., apt install git on Debian/Ubuntu, brew install git on macOS, or Chocolatey on Windows)."
    check_command("git", git_tip)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_subprocess(["git", "clone", repo_url, tmpdir])
        os.chdir(tmpdir)
        if os_sys == "Darwin" or os_sys == "Linux":
            # Check required Unix build tools
            unix_tip = "Install build tools (e.g., apt install build-essential autoconf automake libtool on Debian/Ubuntu; brew install autoconf automake libtool on macOS)."
            for tool in ["autoconf", "automake", "libtool", "make"]:
                check_command(tool, unix_tip)
            
            # Generate build files and compile
            run_subprocess(["./autogen.sh"])
            run_subprocess(["./configure"])
            run_subprocess(["make"])
            lib_dir = ".libs"
            lib_file = "librnnoise.dylib" if os_sys == "Darwin" else "librnnoise.so"
            built_lib = os.path.join(lib_dir, lib_file)
        elif os_sys == "Windows":
            # Check nmake for Visual Studio-based build
            nmake_tip = "Install Visual Studio (community edition) and run this script from Developer Command Prompt. Alternatively, use MSYS2 (install via https://www.msys2.org/) and run make in the repo."
            check_command("nmake", nmake_tip)
            
            os.chdir("msvc")
            # Assumes Visual Studio is installed; run from Developer Command Prompt if needed
            run_subprocess(["nmake", "/f", "Makefile.ms"])
            built_lib = "rnnoise.dll"  # Adjust if name differs after build
        else:
            raise ValueError(f"Unsupported OS: {os_sys}. Manual build required (clone repo and follow README).")
        
        if not os.path.exists(built_lib):
            raise FileNotFoundError(f"Build failed: {built_lib} not found. Ensure build tools are installed and environment is set up (e.g., in containers, add tools via Dockerfile/package manager).")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(script_dir, "files")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(built_lib, os.path.join(target_dir, os.path.basename(built_lib)))

if __name__ == "__main__":
    try:
        build_rnnoise()
        print("RNNoise library built successfully for your OS.")
    except Exception as e:
        print(f"Build failed: {e}")
        print("If in a container/cloud/terminal, ensure tools are installed (e.g., via apt/brew/Chocolatey) or build manually per RNNoise README: https://github.com/xiph/rnnoise")
        sys.exit(1)