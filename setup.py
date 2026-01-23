import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)

setup(
    name="deepfork_cpp",
    version="0.1",
    ext_modules=[CMakeExtension("deepfork_cpp", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)