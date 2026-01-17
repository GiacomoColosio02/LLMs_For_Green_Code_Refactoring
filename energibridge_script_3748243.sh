#!/bin/bash
set -e
cd /tmp/llm_meas_1lwceirs/astropy_97fe6999 && /tmp/llm_meas_1lwceirs/astropy_97fe6999/venv_sweperf/bin/python -m pytest '/tmp/llm_meas_1lwceirs/astropy_97fe6999/astropy/coordinates/tests/test_spectral_coordinate.py::test_create_from_spectral_coord[observer4-target2]' -v
