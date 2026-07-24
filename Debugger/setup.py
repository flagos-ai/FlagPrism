from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


def find_flagtree_source_dir() -> Path:
    configured = os.environ.get("FLAGTREE_SOURCE_DIR")
    if configured:
        source_dir = Path(configured).resolve()
        if (source_dir / "python" / "triton").is_dir():
            return source_dir
        raise RuntimeError(
            f"FLAGTREE_SOURCE_DIR is not a FlagTree source tree: {source_dir}"
        )

    for source_dir in ROOT.parents:
        if (source_dir / "python" / "triton").is_dir() and (
            source_dir / "CMakeLists.txt"
        ).is_file():
            return source_dir
    raise RuntimeError(
        "could not locate the FlagTree source tree; set FLAGTREE_SOURCE_DIR"
    )


def find_libtriton(source_dir: Path) -> Path:
    configured = os.environ.get("FLAGTREE_LIBTRITON")
    if configured:
        path = Path(configured).resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"FLAGTREE_LIBTRITON does not exist: {path}")

    roots = [source_dir / "python", *(Path(item) for item in sys.path if item)]
    patterns = ("libtriton*.so", "libtriton*.dylib", "libtriton*.pyd")
    for root in roots:
        extension_dir = root / "triton" / "_C"
        for pattern in patterns:
            matches = sorted(extension_dir.glob(pattern))
            if matches:
                return matches[0].resolve()
    raise RuntimeError(
        "could not locate triton/_C/libtriton; install FlagTree or set "
        "FLAGTREE_LIBTRITON"
    )


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        extdir.mkdir(parents=True, exist_ok=True)

        source_dir = find_flagtree_source_dir()
        libtriton_path = find_libtriton(source_dir)
        build_dir = Path(
            os.environ.get("FLAGTREE_BUILD_DIR", self.build_temp)
        ).resolve()
        component_build_dir = Path(
            os.environ.get(
                "FLAGTREE_COMPONENT_BUILD_DIR",
                str(Path(self.build_temp).resolve() / "debugger-component"),
            )
        ).resolve()
        component_build_dir.mkdir(parents=True, exist_ok=True)

        llvm_root = os.environ.get("LLVM_SYSPATH", "")
        llvm_dir = os.environ.get(
            "LLVM_DIR", str(Path(llvm_root) / "lib/cmake/llvm") if llvm_root else ""
        )
        mlir_dir = os.environ.get(
            "MLIR_DIR", str(Path(llvm_root) / "lib/cmake/mlir") if llvm_root else ""
        )
        if not llvm_dir or not mlir_dir:
            raise RuntimeError(
                "building flagtree-debugger requires LLVM_DIR and MLIR_DIR, or LLVM_SYSPATH"
            )

        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("building flagtree-debugger requires CMake >= 3.20")
        configure = [
            cmake,
            "-S",
            str(ROOT / "native"),
            "-B",
            str(component_build_dir),
            "-G",
            "Ninja",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DLLVM_DIR={llvm_dir}",
            f"-DMLIR_DIR={mlir_dir}",
            f"-DFLAGTREE_SOURCE_DIR={source_dir}",
            f"-DFLAGTREE_BUILD_DIR={build_dir}",
            f"-DFLAGTREE_LIBTRITON={libtriton_path}",
            f"-DPYTHON_EXTENSION_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX') or '.so'}",
        ]
        llvm_cxx = Path(llvm_root) / "bin/clang++" if llvm_root else None
        if llvm_cxx is not None and llvm_cxx.is_file():
            configure.append(f"-DCMAKE_CXX_COMPILER={llvm_cxx}")
        subprocess.check_call(configure)
        subprocess.check_call(
            [cmake, "--build", str(component_build_dir), "--target", "flagtree_debugger_native"]
        )


build_native = os.environ.get("FLAGTREE_COMPONENT_BUILD_NATIVE", "1") != "0"
setup(
    ext_modules=[CMakeExtension("flagtree_debugger._native")] if build_native else [],
    cmdclass={"build_ext": CMakeBuild} if build_native else {},
    zip_safe=False,
)
