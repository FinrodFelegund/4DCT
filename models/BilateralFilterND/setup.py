from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='bilateralfilterND_torch',
    version='1.0.0',
    author='w/e',
    description='Trainable bilateral filter with spatial, temporal and range optimization',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='w/e',
    py_modules=['bilateral_filter_layer_nd', 'gradcheck'],
    ext_modules=[
        CUDAExtension('bilateral_filter_layer_nd_gpu_lib', [
            'csrc/bilateralfilter_gpu.cu',
            'csrc/bf_layer_gpu_forward.cu',
            'csrc/bf_layer_gpu_backward.cu',
        ],
        include_dirs=['utils', 'csrc'],),
    ],
    cmdclass={'build_ext': BuildExtension}
)