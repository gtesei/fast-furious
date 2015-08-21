#!/bin/bash

echo ">> updating manuals ... "
git pull origin master 
Rscript -e 'library(roxygen2);roxygen2::roxygenise(package.dir = "R-package")'
if [[ $? -eq 0 ]]; then 
      echo ">> checking package ..."
      R CMD check R-package --as-cran
      #R CMD check R-package
fi 