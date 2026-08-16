from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from .operator_test_utils import debugger, require_ascend_debugger


@pytest.fixture
def debug_session(tmp_path: Path):
    require_ascend_debugger()

    @contextmanager
    def start(
        *,
        level: int = 1,
        addr_level: int = 1,
        record_capacity: int = 4096,
    ) -> Iterator[Path]:
        if debugger.is_active():
            debugger.deactivate()
        debugger.clear_exported_runs()
        debugger.reset_config()
        debugger.activate(
            level=level,
            addr_level=addr_level,
            output_dir=tmp_path,
            record_capacity=record_capacity,
            export_raw_records=False,
        )
        try:
            yield tmp_path
        finally:
            if debugger.is_active():
                debugger.deactivate()
            debugger.clear_exported_runs()
            debugger.reset_config()

    return start
