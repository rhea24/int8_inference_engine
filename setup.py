from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="int8_engine",
    ext_modules=[
        CUDAExtension(
            name="int8_engine",
            sources=['int8_matmul.cu', 'bindings.cpp'],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-gencode=arch=compute_80,code=sm_80', # Target A100
                    '-O3' # standard optimizations
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)