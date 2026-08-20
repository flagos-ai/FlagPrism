import triton
import pytest
import subprocess
import json
import pathlib


def test_help():
    # Only check if the viewer can be invoked
    subprocess.check_call(["flagtree-profiler", "-h"],
                          stdout=subprocess.DEVNULL)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path: pathlib.Path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file.with_suffix(""))
    if mode == "script":
        subprocess.check_call(
            ["flagtree-profiler", "-n", name, helper_file, "test"],
            stdout=subprocess.DEVNULL)
    elif mode == "python":
        subprocess.check_call([
            "python3", "-m", "flagtree.profiler.cli", "-n", name, helper_file,
            "test"
        ],
                              stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        subprocess.check_call([
            "flagtree-profiler", "-n", name, "pytest", "-k", "test_main",
            helper_file
        ],
                              stdout=subprocess.DEVNULL)
    with temp_file.open() as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    # Backends may expose allocation/initialization kernels in addition to the
    # profiled kernel and user scope.
    assert len(kernels) >= 2
    assert any(node["frame"]["name"] == "test" for node in kernels)
