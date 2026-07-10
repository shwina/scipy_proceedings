import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; BLUE="#2b6cb0"; LBLUE="#dbe7f4"; GREEN="#4c8c00"; LGREEN="#e4f2d0"; RED="#b23b2e"; GREY="#9aa8ad"
OUT="/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"
AXMAX=215.0  # µs, shared scale

eager=[("cp.abs  (elementwise)",150.2),("CUB reduce",49.9),("reduce final",2.9)]
fused=[("reduce (|x| fused)",28.6),("reduce final",2.7)]

def draw_lane(ax,y,blocks,fc,ec,gap=3.0,label_dur=True,show_names=True):
    t=0
    for name,dur in blocks:
        w=max(dur,3.0)
        ax.add_patch(Rectangle((t,y),w,0.62,fc=fc,ec=ec,lw=1.8))
        if show_names:
            ax.text(t+w/2,y+0.31,name if dur>8 else "",ha="center",va="center",fontsize=9.8,color=INK)
        if label_dur: ax.text(t+w/2,y-0.12,f"{dur:.0f}µs",ha="center",va="top",fontsize=9.1,color="#5a6b70")
        t+=w+gap
    return t

# ---- Figure 1: eager only (slide 4.1) ----
fig,ax=plt.subplots(figsize=(10,1.7)); ax.set_xlim(-4,AXMAX); ax.set_ylim(-0.5,1.0); ax.axis("off")
ax.set_title("Eager  |x| → sum   ·   3 kernels, 203 µs   (RTX 6000 Ada, nsys)",fontsize=14.4,fontweight="bold",color=RED,loc="left")
end=draw_lane(ax,0.15,eager,LBLUE,BLUE)
ax.annotate("intermediate y: 80 MB written → read back",xy=(150,0.95),ha="left",fontsize=10.2,color=RED,style="italic")
ax.plot([-4,AXMAX],[-0.02,-0.02],color=GREY,lw=0.8); ax.text(-4,-0.34,"GPU stream →  (µs)",fontsize=9.6,color="#7c8b90")
fig.savefig(f"{OUT}/timeline_eager.png"); plt.close(fig)

# ---- Figure 2: eager vs fused (slide 4.3) ----
fig,ax=plt.subplots(figsize=(10,2.6)); ax.set_xlim(-4,AXMAX); ax.set_ylim(-0.6,2.1); ax.axis("off")
ax.set_title("Same result, fused: 3 kernels / 203 µs  →  2 kernels / 31 µs  (~6.5×)  — RTX 6000 Ada",
             fontsize=13.8,fontweight="bold",color=INK,loc="left")
ax.text(-4,1.55,"eager",fontsize=10.8,color=RED,fontweight="bold")
draw_lane(ax,1.15,eager,LBLUE,BLUE)
ax.text(-4,0.62,"fused",fontsize=10.8,color=GREEN,fontweight="bold")
draw_lane(ax,0.22,fused,LGREEN,GREEN)
ax.annotate("the 150 µs abs pass + its DRAM round-trip disappear",xy=(70,0.05),ha="left",fontsize=10.3,color=GREEN,style="italic")
fig.savefig(f"{OUT}/timeline_fused.png"); plt.close(fig)
print("wrote timeline_eager.png, timeline_fused.png")

# ---- torch + 3-way comparison ----
torch_k=[("triton abs+sum (fused)",25.9),("triton final",1.5)]
LPURP="#ece3f5"; PURP="#7c3aed"
fig,ax=plt.subplots(figsize=(10,1.7)); ax.set_xlim(-4,AXMAX); ax.set_ylim(-0.5,1.0); ax.axis("off")
ax.set_title("torch.compile  |x| → sum   ·   2 Triton kernels, 27 µs   (implicit fusion)",fontsize=14.4,fontweight="bold",color=PURP,loc="left")
draw_lane(ax,0.15,torch_k,LPURP,PURP)
ax.plot([-4,AXMAX],[-0.02,-0.02],color=GREY,lw=0.8); ax.text(-4,-0.34,"GPU stream →  (µs)",fontsize=9.6,color="#7c8b90")
fig.savefig(f"{OUT}/timeline_torch.png"); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,3.9)); ax.set_xlim(-4,AXMAX); ax.set_ylim(-0.4,4.3); ax.axis("off")
ax.set_title("|x| \u2192 sum on RTX 6000 Ada  \u2014  fusion collapses 3 kernels / 203 \u00b5s \u2192 2 / ~30 \u00b5s",fontsize=13.8,fontweight="bold",color=INK,loc="left")
def lane_lbl(y,txt,c): ax.text(-4,y+0.70,txt,fontsize=11.4,color=c,fontweight="bold")
lane_lbl(3.0,"eager  (CuPy) \u00b7 3 kernels \u00b7 203 \u00b5s",RED);          draw_lane(ax,3.0,eager,LBLUE,BLUE,show_names=True)
lane_lbl(1.6,"torch.compile \u00b7 2 Triton \u00b7 27 \u00b5s",PURP);          draw_lane(ax,1.6,torch_k,LPURP,PURP,show_names=False)
lane_lbl(0.2,"cuda.compute \u00b7 2 CUB \u00b7 31 \u00b5s",GREEN);             draw_lane(ax,0.2,fused,LGREEN,GREEN,show_names=False)
fig.savefig(f"{OUT}/timeline_compare.png"); plt.close(fig)
print("wrote timeline_torch.png, timeline_compare.png")
