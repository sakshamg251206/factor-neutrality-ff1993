PY := .venv/bin/python

.PHONY: all setup results test notebooks paper clean
all: test results notebooks paper

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

results:            ## download data (first run) and regenerate every table, figure and macro
	$(PY) scripts/run_all.py

test:
	.venv/bin/pytest -q

notebooks: results
	cd notebooks && for nb in *.ipynb; do ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace $$nb; done

paper: results
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
	pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cp paper/main.pdf paper/Factor_Neutrality_FF1993_Garg_2026.pdf

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
