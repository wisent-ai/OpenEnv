TEX      = pdflatex
BIB      = bibtex
MAIN     = main
OUTDIR   = .

.PHONY: all clean view slides

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex figures/*.tex
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).pdf $(MAIN).synctex.gz \
	      sections/*.aux figures/*.aux

view: $(MAIN).pdf
	open $(MAIN).pdf

slides: slides.pdf
slides.pdf: slides.tex figures/*.tex
	$(TEX) slides
	$(TEX) slides
