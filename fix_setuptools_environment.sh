#!/bin/bash
# Temporarily downgrade setuptools for old repos
pip install --break-system-packages 'setuptools<58.0.0' --force-reinstall
