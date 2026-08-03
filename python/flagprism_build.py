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
    names = (
        "TRITON_BUILD_FLAGPRISM",
        "TRITON_BUILD_DEVTOOLS",
        "TRITON_BUILD_PROTON",
    )
    values = {_check_env_flag(name) for name in names if name in os.environ}
    if len(values) > 1:
        raise RuntimeError(
            "FlagPrism components cannot be enabled independently. Set "
            "TRITON_BUILD_FLAGPRISM=ON for the combined tools build or OFF "
            "for a core-only build."
        )
    return values.pop() if values else default


@dataclass(frozen=True)
class FlagPrismBuildConfig:
    enabled: bool
    relative_root: Path
    root: Path

    @classmethod
    def from_environment(cls, project_root: Path) -> "FlagPrismBuildConfig":
        relative_root = Path("third_party") / "FlagPrism"
        return cls(
            enabled=_flagprism_enabled(),
            relative_root=relative_root,
            root=project_root / relative_root,
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
                self.root / "proton" / "CMakeLists.txt",
                self.root / "proton" / "proton" / "__init__.py",
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
        if self.enabled:
            args.extend([
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

        expected_native = "libproton" + (
            sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        )
        for artifact in (triton_root / "_C").glob("libproton*"):
            if artifact.name != expected_native:
                artifact.unlink(missing_ok=True)

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
                str(self.relative_root / "proton" / "proton"),
            ),
            (
                "flagtree.profiler.hooks",
                str(self.relative_root / "proton" / "proton" / "hooks"),
            ),
        )

    def packages(self) -> tuple[str, ...]:
        return tuple(package for package, _ in self.package_dirs())

    def console_scripts(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            "proton = flagtree.profiler.proton:main",
            "proton-viewer = flagtree.profiler.viewer:main",
        ]


def create_build_config(project_root: Path) -> FlagPrismBuildConfig:
    config = FlagPrismBuildConfig.from_environment(project_root)
    config.validate_sources()
    return config
