from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='bilateralfilter4D_torch_lut',
    version='1.0.0',
    author='w/e',
    author_email='w/e',
    description='Trainable bilateral filter layer with trainable 4D kernel and 2D intensity lookup table',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='w/e',
    py_modules=['bilateral_filter_layer_4d_lut', 'gradcheck'],
    ext_modules=[
        CUDAExtension('bilateralfilter4dlut_gpu_lib', [
            'csrc/bilateralfilter_gpu.cu',
            'csrc/bf_layer_gpu_forward.cu',
            'csrc/bf_layer_gpu_backward.cu',
        ],
            include_dirs=['utils', 'csrc'],),
    ],
    cmdclass={'build_ext': BuildExtension}
)