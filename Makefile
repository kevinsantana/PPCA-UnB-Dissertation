current_dir := $(shell pwd)
clean:
	./scripts/clean_tex_build_files.sh

build:
	pdflatex dissertation_main.tex

references:
	pdflatex dissertation_main.tex
	bibtex dissertation_main
	makeglossaries dissertation_main
	pdflatex dissertation_main.tex
