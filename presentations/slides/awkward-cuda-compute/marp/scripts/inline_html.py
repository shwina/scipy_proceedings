#!/usr/bin/env python3
"""Inline figs/*.png into a Marp HTML file (in place) so it's a single portable file."""
import re, base64, os, sys

path = sys.argv[1]
html = open(path).read()

def repl(m):
    p = m.group(1)
    if os.path.exists(p):
        b = base64.b64encode(open(p, "rb").read()).decode()
        return f'src="data:image/png;base64,{b}"'
    return m.group(0)

out = re.sub(r'src="(figs/[^"]+\.png)"', repl, html)
open(path, "w").write(out)
print(f"inlined {out.count('data:image/png;base64')} image(s) into {path}")
