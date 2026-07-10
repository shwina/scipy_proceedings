import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
HERE=os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from deck_config import data as _data
d=json.load(open(_data("bench_ak_argmin.json")))
INK="#17303a"; RED="#b23b2e"; LRED="#f6d9d5"; GREEN="#4c8c00"; LGREEN="#dcedc3"
rows=[("handwritten\nCUDA kernel", d["before_ms"], RED, LRED),
      ("cuda.compute\nbackend",      d["after_ms"],  GREEN, LGREEN)]
fig,ax=plt.subplots(figsize=(9.2,2.9))
y=[1,0]
for yi,(lbl,val,ec,fc) in zip(y,rows):
    ax.barh(yi,val,color=fc,edgecolor=ec,lw=2.2,height=0.58,zorder=3)
    ax.text(val*1.03,yi,f"  {val:.2f} ms",va="center",ha="left",fontsize=15.6,fontweight="bold",color=ec)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=14.4)
ax.set_xlim(0, max(d["before_ms"],d["after_ms"])*1.35)
ax.set_xlabel("time per ak.argmin call  (ms)", fontsize=12.6)
ax.set_title(f"ak.argmin over ragged data · {d['num_segments']:,} sublists · {d['device']}",
             fontsize=14.4, fontweight="bold", color=INK, loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(left=False); ax.grid(axis="x", ls=":", color="#c9d2d6", zorder=0)
spd=d["before_ms"]/d["after_ms"]
tag = f"{spd:.1f}x faster" if spd>=1 else f"{1/spd:.1f}x slower"
ax.text(0.99,0.06,f"cuda.compute backend: {tag}   ·   same result",
        transform=ax.transAxes, ha="right", fontsize=12.6, color=GREEN, fontweight="bold")
fig.savefig(f"{HERE}/../figs/bench_ak_argmin.png"); plt.close(fig)
print("wrote figs/bench_ak_argmin.png")
