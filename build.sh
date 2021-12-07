#!/usr/bin/env bash

set -e
quietmode=true
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd scripts

echo "Building Artifacts, this may take a second"

cd ..

# bazel build ee510final
echo "Building PDF document"
if [ "$quietmode" = true ] ; then
    python3 Main.py 
    pdflatex -shell-escape -halt-on-error Final.tex > /dev/null 2>&1
    pdflatex -shell-escape -halt-on-error Final.tex > /dev/null 2>&1
else
    python3 Main.py
    pdflatex -shell-escape -halt-on-error Final.tex 
    pdflatex -shell-escape -halt-on-error Final.tex
fi
