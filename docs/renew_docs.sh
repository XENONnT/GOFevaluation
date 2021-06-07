#!/bin/bash

sphinx-apidoc -o source/ ../GOFevaluation -f
make html
