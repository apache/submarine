#!/bin/bash
printf "Building Submarine CI/CD Image.\n"
docker build -t submarine-cicd .
printf "Start Submarine CI/CD.\n"
docker run -it --rm submarine-cicd
