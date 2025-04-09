#!/bin/bash

find . -type f \( \
    -name "*.acn" -or \
    -name "*.aux" -or \
    -name "*.bbl" -or \
    -name "*.blg" -or \
    -name "*.brf" -or \
    -name "*.glo" -or \
    -name "*.glsdefs" -or \
    -name "*.ist" -or \
    -name "*.lof" -or \
    -name "*.log" -or \
    -name "*.lot" -or \
    -name "*.out" -or \
    -name "*.toc" -or \
    -name "*.acr" -or \
    -name "*.alg" -or \
    -name "*.bbl" -or \
    -name "*.blg" -or \
    -name "*.brf" -or \
    -name "*.glg" -or \
    -name "*.gls" -or \
    -name "*.fdb_latexmk" -or \
    -name "*.fuse_*" \
\) -print0 | xargs -0 rm -f
