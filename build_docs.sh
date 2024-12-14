cd docs
make clean
rm -rf source test
sphinx-apidoc -o source ../src
rm source/modules.rst # if separating the modules into different rst then don't remove it and update its content to get the corresponding rst
sphinx-apidoc -o test ../test
rm test/modules.rst
make html
firefox _build/html/index.html