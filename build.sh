#!/usr/bin/env bash

set -e
quietmode=false
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" failed: command filed with exit code $?."' EXIT


if [ "$quietmode" = true ] ; then
    echo "Building Artifacts, this may take a second"
    cd scripts
    python3 Main.py > /dev/null 2>&1
    cd ..
    echo "Building PDF"
    pdflatex -shell-escape Final.tex > /dev/null 2>&1
    pdflatex -shell-escape Final.tex > /dev/null 2>&1
else
    echo "Building Artifacts, this may take a second"
    cd scripts
    python3 Main.py 
    cd ..
    echo "Building PDF"
    pdflatex -shell-escape Final.tex 
    pdflatex -shell-escape Final.tex
fi
echo "Success"
