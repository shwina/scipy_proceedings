# Build the GPU-Accelerated Awkward Arrays deck (Marp).
#
#   make pdf    -> slides.pdf   (presentation PDF, code line numbers)
#   make html   -> slides.html  (self-contained single file; presenter view + notes via 'p')
#   make        -> both
#   make clean  -> remove outputs
#
# Code line numbers are added by marp.config.js -> scripts/marp-lineno.js (auto-loaded).

MARP ?= marp
PY   ?= python3
SRC  := slides.md
DEPS := $(SRC) marp.config.js scripts/marp-lineno.js $(wildcard figs/*.png)

# Headless Chrome for PDF rendering. This dev box needs a --no-sandbox wrapper;
# it is exported only if present, so a normal Chrome install elsewhere still works.
CHROME_PATH ?= /home/coder/.local/chrome/chrome-nosandbox.sh
ifneq ($(wildcard $(CHROME_PATH)),)
export CHROME_PATH
endif

FLAGS := --no-stdin --html --allow-local-files

.PHONY: all pdf html clean
all: pdf html

pdf: slides.pdf
slides.pdf: $(DEPS)
	$(MARP) $(FLAGS) --browser chrome --pdf $(SRC) -o $@

html: slides.html
slides.html: $(DEPS)
	$(MARP) $(FLAGS) $(SRC) -o $@
	$(PY) scripts/inline_html.py $@

clean:
	rm -f slides.pdf slides.html
