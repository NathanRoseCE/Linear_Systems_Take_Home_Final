#!/usr/bin/env bash

set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd scripts

python3 Main.py

cd ..

# bazel build ee510final
pdflatex -shell-escape Final.tex
pdflatex -shell-escape Final.tex
