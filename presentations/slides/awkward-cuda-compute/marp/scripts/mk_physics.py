import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; RED="#b23b2e"; LRED="#f6d9d5"; GREEN="#4c8c00"; LGREEN="#dcedc3"; GREY="#9aa8ad"
OUT="figs"
fig,ax=plt.subplots(figsize=(10,3.4)); ax.set_xlim(0,100); ax.set_ylim(-0.3,2.5); ax.axis("off")
ax.set_title("Real analysis: select di-muon events, then invariant mass  (Nsight, RTX 6000 Ada)",
             fontsize=15.0,fontweight="bold",color=INK,loc="left")

# ---- today: ~260 tiny kernels ----
ax.text(0,2.05,"Awkward today",fontsize=13.2,fontweight="bold",color=RED)
ax.text(100,2.05,"~260 tiny kernel launches  (CuPy + raw CUDA C++)",fontsize=12.0,color=RED,ha="right")
import math
n=64; x=0.0; w=100.0/n
for i in range(n):
    ww=w*0.62
    ax.add_patch(Rectangle((x,1.35),ww,0.45,fc=LRED,ec=RED,lw=0.7))
    x+=w

# ---- tomorrow: ~30 fused kernels ----
ax.text(0,0.85,"with cuda.compute",fontsize=13.2,fontweight="bold",color=GREEN)
ax.text(100,0.85,"~30 fused kernels  ·  3x faster",fontsize=12.0,color=GREEN,ha="right")
# fewer, wider blocks grouped into 3 sections
groups=[(0,10),(14,9),(40,6)]
gi=0
xs=0
for start,cnt in [(2,8),(40,7),(74,5)]:
    x=start
    for i in range(cnt):
        ax.add_patch(Rectangle((x,0.15),3.2,0.45,fc=LGREEN,ec=GREEN,lw=1.0))
        x+=3.9
# section brackets
for (s,e,lbl) in [(1,34,"select muons"),(39,66,"exactly 2 muons"),(73,95,"invariant mass")]:
    ax.plot([s,e],[-0.05,-0.05],color=GREY,lw=1)
    ax.text((s+e)/2,-0.22,lbl,ha="center",fontsize=10.2,color="#5a6b70")
fig.savefig(f"{OUT}/physics_analysis.png"); plt.close(fig)
print("wrote physics_analysis.png")
