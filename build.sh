#!/usr/bin/env bash

set -e

make='make -j17'

quietmode=false
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" failed: command filed with exit code $?."' EXIT

python3 tests/testMain.py


if [ "$quietmode" = true ] ; then
    echo "Building Artifacts, this may take a second"
    ${make} > /dev/null 2>&1
else
    echo "Building Artifacts, this may take a second"
    ${make}
fi
echo "Complete"
