from distutils.core import setup, Extension

module = Extension("micro_torch", sources=["micro_torch.c"])

setup(name="PackageName", version="1.0", description="This is a package for my module.", ext_modules=[module])