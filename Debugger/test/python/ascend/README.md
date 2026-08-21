# Ascend debugger CI tests

This directory contains a standalone Triton test set for the FlagPrism
debugger.  It covers 30 representative operator patterns and 10 runtime metric
cases.  FlagGems is not required.

Run the complete set on an Ascend FlagTree build with:

```bash
python3 -m pytest -q \
  third_party/FlagPrism/Debugger/test/python/ascend \
  -m ascend_debugger_ci
```

The suite intentionally does not issue a physically invalid device access.
Address cases validate contiguous and prefix-masked load/store summaries without
risking the NPU context used by later CI jobs.
