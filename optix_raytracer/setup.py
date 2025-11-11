"""Setup script for installing optix_voxel_tracer Python module."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
from pathlib import Path


class CMakeBuild(build_ext):
    """Custom build extension that uses CMake."""

    def run(self):
        """Run CMake build."""
        try:
            import subprocess
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """Build extension using CMake."""
        import subprocess
        import shutil

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # CMake build directory
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Check for OptiX SDK
        optix_dir = os.environ.get('OptiX_INSTALL_DIR')
        if not optix_dir:
            raise RuntimeError(
                "OptiX_INSTALL_DIR environment variable not set. "
                "Please set it to your OptiX SDK installation directory."
            )

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DOptiX_INSTALL_DIR={optix_dir}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        build_args = ['--config', 'Release']

        # Build
        subprocess.check_call(
            ['cmake', str(Path(__file__).parent)] + cmake_args,
            cwd=build_temp
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )


class CMakeExtension(Extension):
    """CMake extension - doesn't need sources."""

    def __init__(self, name):
        super().__init__(name, sources=[])


setup(
    name='optix-voxel-tracer',
    version='0.1.0',
    author='Your Name',
    description='OptiX-based ray tracer for voxel grids',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('optix_voxel_tracer')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.20.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
    ],
)
