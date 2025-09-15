"""
Custom setup.py for cuBLASter Cython extensions with CUDA support.

This setup.py handles the compilation of CUDA code (.cu files) along with
Cython extensions, providing cuBLAS-accelerated matrix operations.
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


class CudaExtension(Extension):
    """Extension class for CUDA code compilation"""
    pass


class BuildExtension(build_ext):
    """Custom build extension to handle CUDA compilation"""
    
    def build_extensions(self):
        # Find CUDA installation
        cuda_home = self._find_cuda()
        if not cuda_home:
            raise RuntimeError("CUDA installation not found. Please install CUDA toolkit.")
        
        print(f"Found CUDA at: {cuda_home}")
        
        # Update compiler settings for CUDA extensions
        for ext in self.extensions:
            if isinstance(ext, CudaExtension):
                self._configure_cuda_extension(ext, cuda_home)
        
        # Call parent build
        build_ext.build_extensions(self)
    
    def _find_cuda(self):
        """Find CUDA installation directory"""
        # Check environment variable
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home and os.path.exists(cuda_home):
            return cuda_home
        
        # Check common installation paths
        common_paths = [
            '/usr/local/cuda',
            '/opt/cuda',
            '/usr/local/cuda-12.3',
            '/usr/local/cuda-12.2',
            '/usr/local/cuda-12.1',
            '/usr/local/cuda-12.0',
            '/usr/local/cuda-11.8',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find nvcc in PATH
        try:
            nvcc_path = subprocess.check_output(['which', 'nvcc'], stderr=subprocess.DEVNULL).decode().strip()
            if nvcc_path:
                return os.path.dirname(os.path.dirname(nvcc_path))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def _configure_cuda_extension(self, ext, cuda_home):
        """Configure extension for CUDA compilation"""
        # Add CUDA include directories
        ext.include_dirs.extend([
            os.path.join(cuda_home, 'include'),
            np.get_include()
        ])
        
        # Add CUDA library directories
        if sys.platform.startswith('win'):
            ext.library_dirs.extend([
                os.path.join(cuda_home, 'lib', 'x64'),
                os.path.join(cuda_home, 'lib')
            ])
        else:
            ext.library_dirs.extend([
                os.path.join(cuda_home, 'lib64'),
                os.path.join(cuda_home, 'lib')
            ])
        
        # Add CUDA libraries
        ext.libraries.extend(['cudart', 'cublas'])
        
        # Add compiler flags
        if sys.platform.startswith('win'):
            ext.extra_compile_args = ['/std:c++17']
        else:
            ext.extra_compile_args = ['-std=c++17']
        
        # CUDA-specific compilation
        self._compile_cuda_sources(ext, cuda_home)
    
    def _compile_cuda_sources(self, ext, cuda_home):
        """Compile CUDA source files"""
        nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
        if sys.platform.startswith('win'):
            nvcc += '.exe'
        
        if not os.path.exists(nvcc):
            raise RuntimeError(f"nvcc not found at {nvcc}")
        
        # Find .cu files in sources
        cu_sources = [src for src in ext.sources if src.endswith('.cu')]
        cpp_sources = [src for src in ext.sources if not src.endswith('.cu')]
        
        # Compile .cu files to .o files
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        compiled_objects = []
        for cu_file in cu_sources:
            obj_file = os.path.join(build_temp, os.path.basename(cu_file) + '.o')
            
            nvcc_cmd = [
                nvcc,
                '-c', cu_file,
                '-o', obj_file,
                '-std=c++17',
                '--compiler-options', '-fPIC' if not sys.platform.startswith('win') else '/MD',
                '-I' + np.get_include(),
                '--expt-relaxed-constexpr'
            ]
            
            # Add include directories
            for inc_dir in ext.include_dirs:
                nvcc_cmd.extend(['-I', inc_dir])
            
            print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
            subprocess.check_call(nvcc_cmd)
            compiled_objects.append(obj_file)
        
        # Add compiled objects to extension
        ext.extra_objects.extend(compiled_objects)
        # Remove .cu files from sources to avoid double compilation
        ext.sources = cpp_sources


def get_extensions():
    """Define extensions to build"""
    
    # Try to detect CUDA availability
    cuda_available = False
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, timeout=10)
        cuda_available = result.returncode == 0
        print(f"CUDA detected: {cuda_available}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("CUDA not detected, using CPU fallback")
    
    if cuda_available:
        # Use CUDA version
        extensions = [
            CudaExtension(
                name="cublaster_core",
                sources=[
                    "core/cublaster.pyx",
                    "core/cuda_matmul.cu"
                ],
                include_dirs=[
                    "core",
                    np.get_include()
                ],
                libraries=[],
                library_dirs=[],
                extra_compile_args=[],
                extra_link_args=[],
                extra_objects=[],
                language="c++",
            )
        ]
    else:
        # Use CPU fallback
        extensions = [
            Extension(
                name="cublaster_core",
                sources=[
                    "core/cublaster.pyx",
                    "core/cpu_fallback.cpp"
                ],
                include_dirs=[
                    "core",
                    np.get_include()
                ],
                extra_compile_args=["-std=c++17"],
                language="c++",
            )
        ]
    
    return cythonize(
        extensions,
        language_level="3",
        build_dir="build/cython",
        compiler_directives={"embedsignature": True},
    )


def main():
    """Main setup function"""
    print("Setting up cuBLASter with CUDA support...")
    
    setup(
        name="cublaster",
        version="0.1.0",
        description="CUDA-accelerated lattice reduction library",
        author="cuBLASter Team",
        python_requires=">=3.10",
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        install_requires=[
            "numpy>=1.20.0",
            "cython>=3.0.0",
            "cysignals",
            "setuptools",
            "matplotlib"
        ],
    )


if __name__ == "__main__":
    main()