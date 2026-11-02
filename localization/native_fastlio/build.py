"""Build FAST-LIO in the current Windows Python environment, without ROS."""
import os
from pathlib import Path
import subprocess
import sys
import pybind11
from prepare_source import prepare


def main():
    root=Path(__file__).resolve().parent
    vcpkg=Path(os.environ.get('VCPKG_ROOT',root.parents[1]/'.tools/vcpkg'))
    toolchain=vcpkg/'scripts/buildsystems/vcpkg.cmake'
    if not toolchain.is_file():
        raise SystemExit('Set VCPKG_ROOT to an existing vcpkg checkout')
    prepare()
    configure=[sys.executable,'-m','cmake','-S',str(root),'-B',str(root/'build-windows'),
               f'-DCMAKE_TOOLCHAIN_FILE={toolchain}',f'-DPython_EXECUTABLE={sys.executable}',
               f'-Dpybind11_DIR={pybind11.get_cmake_dir()}']
    if sys.platform=='win32':
        configure.extend(['-A','x64'])
    subprocess.run(configure,check=True)
    subprocess.run([sys.executable,'-m','cmake','--build',str(root/'build-windows'),
                    '--config','Release','--parallel','2'],check=True)


if __name__=='__main__':
    main()
