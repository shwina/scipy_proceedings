import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; GREEN="#4c8c00"; LGREEN="#e4f2d0"; GREY="#9aa8ad"; LGREY="#eef1f2"
OUT="figs"
libs=[
 ("cuda.core",     "Pythonic CUDA runtime access", LGREY, GREY),
 ("cuda.bindings", "Low-level CUDA C/C++ bindings", LGREY, GREY),
 ("cuda.compute",  "Composable parallel algorithms", LGREEN, GREEN),
 ("cuda.tile",     "Tile-based GPU kernels", LGREY, GREY),
 ("nvmath-python", "Pythonic NVIDIA math libraries", LGREY, GREY),
]
fig,ax=plt.subplots(figsize=(12.2,4.6)); ax.set_xlim(0,12.2); ax.set_ylim(0,4.6); ax.axis("off")
# parent
ax.add_patch(FancyBboxPatch((4.6,3.7),3.0,0.7,boxstyle="round,pad=0.02,rounding_size=0.1",fc="#e8eef2",ec=INK,lw=2))
ax.text(6.1,4.05,"CUDA Python",ha="center",va="center",fontsize=16.8,fontweight="bold",color=INK)
n=len(libs); w=2.25; gap=0.15; total=n*w+(n-1)*gap; x0=(12.2-total)/2
for i,(name,blurb,fc,ec) in enumerate(libs):
    x=x0+i*(w+gap); y=0.55; h=2.35
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=2.2))
    ax.text(x+w/2,y+h-0.34,name,ha="center",va="center",fontsize=15.0,fontweight="bold",
            color=INK, family="monospace")
    wrapped="\n".join(textwrap.wrap(blurb,14))
    ax.text(x+w/2,y+0.95,wrapped,ha="center",va="center",fontsize=15.0,color="#2c3b41")
    ax.add_patch(FancyArrowPatch((6.1,3.7),(x+w/2,y+h),arrowstyle="-|>",mutation_scale=12,lw=1.3,color=GREY))
# highlight this talk
ci=2; cx=x0+ci*(w+gap)+w/2
ax.text(cx,0.2,"this talk",ha="center",fontsize=11.4,color=GREEN,fontweight="bold")
fig.savefig(f"{OUT}/ecosystem.png"); plt.close(fig)
print("wrote ecosystem.png")
