current_dir := $(shell pwd)
clean:
	rm -rf *.acn \
	       *.aux \
		   *.bbl \
		   *.blg \
		   *.brf \
	       *.glo \
	       *.glsdefs \
	       *.ist \
		   *.lof \
	       *.log \
		   *.lot \
	       *.out \
	       *.toc \
		   *.acr \
		   *.alg \
		   *.bbl \
		   *.blg \
		   *.brf \
		   *.glg \
		   *.gls \
		   *.fdb_latexmk \
		   *.fuse_*

build:
	pdflatex dissertation_main.tex

references:
	pdflatex dissertation_main.tex
	bibtex dissertation_main
	makeglossaries dissertation_main
	pdflatex dissertation_main.tex
