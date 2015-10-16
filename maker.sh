#!/bin/bash

read -p "Commit description: " desc

echo ">> updating manuals ... "
Rscript -e 'library(roxygen2);roxygen2::roxygenise(package.dir = "R-package")'

if [[ $? -eq 0 ]]; then 
      echo ">> checking package ..."
      R CMD check R-package --as-cran
      #R CMD check R-package
      if [[ $? -eq 0 ]]; then
            echo ">> adding updates ..."
            git status 
            git add R-package/*
            cp R-package.Rcheck/fastfurious-manual.pdf . 
	    git add fastfurious-manual.pdf
            if [[ $? -eq 0 ]]; then
                  echo ">> committing and pushing to github ..."
                  git status 
                  git commit -m "$desc"
                  git push origin master 
            fi    
      fi 
fi 