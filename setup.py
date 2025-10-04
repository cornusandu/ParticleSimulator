from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import subprocess, os, numpy

CUDA_HOME = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc.exe")

nvcc_compile_args = [
    "-O3",
    "-std=c++20",
    "-rdc=true",
    "-arch=sm_61",
    "--compiler-options", "/MD"
]

msvc_compile_args = ["/O2", "/EHsc"]

class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler.compile
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        def _nvcc_compile(sources, output_dir=None, *args, **kwargs):
            cuda_sources = [s for s in sources if s.endswith('.cu')]
            cpp_sources  = [s for s in sources if not s.endswith('.cu')]
            objects = []

            for src in cuda_sources:
                abs_src = os.path.abspath(src).replace("\\", "/")
                obj = os.path.join(build_temp,
                                   os.path.basename(src) + ".obj").replace("\\", "/")
                cmd = [
                    NVCC, "-c", abs_src, "-o", obj,
                    "-O3", "-std=c++20", "-rdc=true", "-arch=sm_61",
                    "--compiler-options", "/MD",
                    "-I", numpy.get_include().replace("\\", "/"),
                    "-I", os.path.join(CUDA_HOME, "include").replace("\\", "/"),
                ]
                print("\nCompiling with NVCC:\n", " ".join(cmd))
                subprocess.check_call(cmd, shell=False)
                objects.append(obj)

            if cpp_sources:
                objects += original_compile(
                    cpp_sources, output_dir=output_dir, *args, **kwargs)
            return objects

        # <-- Replace the compiler method before invoking the parent build.
        self.compiler.compile = _nvcc_compile
        super().build_extensions()

ext = Extension(
    name="compute",
    sources=["compute.pyx", "gpu_compute.cu"],
    include_dirs=[
        numpy.get_include(),
        os.path.join(CUDA_HOME, "include")
    ],
    library_dirs=[os.path.join(CUDA_HOME, "lib", "x64")],
    libraries=["cudart", "cudadevrt"],
    language="c++",
    extra_compile_args={"msvc": msvc_compile_args},
)

setup(
    name="compute",
    ext_modules=cythonize([ext], language_level=3),
    cmdclass={"build_ext": BuildExt},
)
