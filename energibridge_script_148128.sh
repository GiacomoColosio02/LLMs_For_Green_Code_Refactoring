#!/bin/bash
set -e
cd /tmp/tmp_cuwdwcr/requests_90a166d4 && /tmp/tmp_cuwdwcr/requests_90a166d4/venv_sweperf/bin/python -m pytest tests/test_requests.py::TestRequests::test_errors[http://localhost:1-ConnectionError] -v
