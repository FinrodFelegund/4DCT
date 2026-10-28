from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.rst') as f:
    long_description = f.read()


setup(
    name='bilaterfilterND_torch_embedding',
    version='1.0.0',
    author='w/e',
    description='Trainable bilateral filter with spatio-temporal gaussian filter and intensity embedding feed forward',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='w/e',
    py_modules=['bilateral_filter_layer_nd_embedding'],
    ext_modules=[
        CUDAExtension('bilateral_filter_nd_embedding_gpu_lib', [
            'csrc/bilateralfilter_gpu.cu',
            'csrc/bf_layer_gpu_foward.cu',
            'csrc/bf_layer_gpu_backward.cu'
        ],
        include_dirs=['utils', 'csrc'],),
    ],
    cmdclass={'build_ext': BuildExtension}
)