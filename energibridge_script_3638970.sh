#!/bin/bash
set -e
cd /tmp/llm_meas_tzk1rout/astropy_97fe6999 && /tmp/llm_meas_tzk1rout/astropy_97fe6999/venv_sweperf/bin/python -m pytest '/tmp/llm_meas_tzk1rout/astropy_97fe6999/astropy/coordinates/tests/test_sky_coord.py::test_coord_init_unit' -v
