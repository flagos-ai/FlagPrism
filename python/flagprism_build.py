"""Setuptools integration policy for the bundled FlagPrism components."""

from __future__ import annotations

import os
import shutil
import sysconfig
from dataclasses import dataclass
from pathlib import Path


def _check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ("ON", "1", "YES", "TRUE", "Y")


def _flagprism_enabled(default: bool = True) -> bool:
    return _check_env_flag(
        "TRITON_BUILD_FLAGPRISM", "ON" if default else "OFF"
    )


@dataclass(frozen=True)
class FlagPrismBuildConfig:
    enabled: bool
    relative_root: Path
    root: Path

    @classmethod
    def from_environment(
        cls, project_root: Path, source_root: Path | None = None
    ) -> "FlagPrismBuildConfig":
        configured_root = os.getenv("FLAGPRISM_SOURCE_DIR")
        root = Path(configured_root) if configured_root else source_root
        if root is None:
            root = Path("third_party") / "FlagPrism"
        if not root.is_absolute():
            root = project_root / root
        root = root.resolve()
        relative_root = Path(os.path.relpath(root, project_root.resolve()))
        return cls(
            enabled=_flagprism_enabled(),
            relative_root=relative_root,
            root=root,
        )

    def validate_sources(self) -> None:
        required = []
        if self.enabled:
            required.extend((
                self.root / "cmake" / "FlagPrism.cmake",
                self.root / "Debugger" / "native" / "CMakeLists.txt",
                self.root / "Debugger" / "python" / "flagtree_debugger" / "__init__.py",
                self.root / "Debugger" / "python" / "flagtree_debugger" / "language.py",
                self.root / "Debugger" / "python" / "flagtree_debugger" / "statement.py",
                self.root / "Profiler" / "CMakeLists.txt",
                self.root / "Profiler" / "python" / "flagtree_profiler" / "__init__.py",
            ))
        missing = [
            str(self.relative_root / path.relative_to(self.root))
            for path in required
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "FlagPrism sources are missing. Initialize the submodule "
                "with `git submodule update --init --recursive`. Missing: "
                + ", ".join(missing)
            )

    def cmake_args(self, build_lib: str) -> list[str]:
        args = [
            "-DTRITON_BUILD_FLAGPRISM=" + ("ON" if self.enabled else "OFF"),
        ]
        backend = os.getenv("FLAGPRISM_BACKEND", "").strip()
        if backend:
            args.append("-DFLAGPRISM_BACKEND=" + backend)
        if self.enabled:
            args.extend([
                "-DFLAGPRISM_SOURCE_DIR=" + str(self.root),
                "-DFLAGPRISM_PYTHON_DIR=" + os.path.abspath(build_lib),
                "-DPYTHON_EXTENSION_SUFFIX=" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"),
            ])
        return args

    def prepare_build_tree(self, build_lib: str) -> None:
        build_root = Path(build_lib)
        triton_root = build_root / "triton"
        flagtree_root = build_root / "flagtree"

        # A reused setuptools tree may contain packages from the former split
        # wheels. Remove them before CMake writes the current native outputs.
        for package in ("debugger", "profiler"):
            shutil.rmtree(triton_root / package, ignore_errors=True)
        for package in ("flagtree_debugger", "flagtree_profiler"):
            shutil.rmtree(build_root / package, ignore_errors=True)
        for package in ("debugger", "profiler"):
            shutil.rmtree(flagtree_root / package, ignore_errors=True)
        for module in ("_components.py", "_devtools.py", "_statement_metadata.py"):
            (triton_root / module).unlink(missing_ok=True)
        for module in ("_components", "_devtools", "_statement_metadata"):
            for artifact in (triton_root / "__pycache__").glob(f"{module}.*.pyc"):
                artifact.unlink(missing_ok=True)
        for artifact in (triton_root / "_C").glob("libproton*"):
            artifact.unlink(missing_ok=True)

    def finalize_build_tree(self, build_lib: str) -> None:
        """Remove source-tree artifacts copied after the native build."""
        build_root = Path(build_lib)
        triton_root = build_root / "triton"
        flagtree_root = build_root / "flagtree"

        for package in ("debugger", "profiler"):
            shutil.rmtree(triton_root / package, ignore_errors=True)
        for package in ("flagtree_debugger", "flagtree_profiler"):
            shutil.rmtree(build_root / package, ignore_errors=True)
        for module in ("_components.py", "_devtools.py", "_statement_metadata.py"):
            (triton_root / module).unlink(missing_ok=True)
        for module in ("_components", "_devtools", "_statement_metadata"):
            for artifact in (triton_root / "__pycache__").glob(f"{module}.*.pyc"):
                artifact.unlink(missing_ok=True)

        if not self.enabled:
            shutil.rmtree(flagtree_root / "debugger", ignore_errors=True)
            shutil.rmtree(flagtree_root / "profiler", ignore_errors=True)
            for artifact in (triton_root / "_C").glob("libproton*"):
                artifact.unlink(missing_ok=True)
            return

        for artifact in (triton_root / "_C").glob("libproton*"):
            artifact.unlink(missing_ok=True)

        expected_native = "_native" + (
            sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        )
        native_path = flagtree_root / "profiler" / expected_native
        if not native_path.is_file():
            raise RuntimeError(
                f"FlagTree Profiler native module was not built: {native_path}"
            )

    def package_dirs(self) -> tuple[tuple[str, str], ...]:
        if not self.enabled:
            return ()
        return (
            (
                "flagtree.debugger",
                str(self.relative_root / "Debugger" / "python" / "flagtree_debugger"),
            ),
            (
                "flagtree.profiler",
                str(self.relative_root / "Profiler" / "python" / "flagtree_profiler"),
            ),
            (
                "flagtree.profiler.hooks",
                str(self.relative_root / "Profiler" / "python" / "flagtree_profiler" / "hooks"),
            ),
        )

    def packages(self) -> tuple[str, ...]:
        return tuple(package for package, _ in self.package_dirs())

    def console_scripts(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            "flagtree-profiler = flagtree.profiler.cli:main",
            "flagtree-profiler-viewer = flagtree.profiler.viewer:main",
        ]


def create_build_config(
    project_root: Path, source_root: Path | None = None
) -> FlagPrismBuildConfig:
    config = FlagPrismBuildConfig.from_environment(project_root, source_root)
    config.validate_sources()
    return config
