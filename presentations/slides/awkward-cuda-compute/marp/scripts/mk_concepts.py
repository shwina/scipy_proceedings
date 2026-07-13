import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; GREEN="#4c8c00"; LGREEN="#e4f2d0"; BLUE="#2b6cb0"; LBLUE="#dbe7f4"
GREY="#9aa8ad"; LGREY="#eef1f2"; RED="#b23b2e"; LRED="#fbe9e6"; PURP="#7c3aed"
OUT="/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"
def box(ax,x,y,w,h,title,sub=None,fc=LGREY,ec=GREY,fs=14.4,tc=INK,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.09",fc=fc,ec=ec,lw=2))
    ax.text(x+w/2,y+h/2+(0.11 if sub else 0),title,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",color=tc)
    if sub: ax.text(x+w/2,y+h/2-0.25,sub,ha="center",va="center",fontsize=10.8,color="#5a6b70")
def arr(ax,p1,p2,c=INK,lw=2.2):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=15,lw=lw,color=c))

# 1.1 ecosystem
fig,ax=plt.subplots(figsize=(10,3.4)); ax.set_xlim(0,10); ax.set_ylim(0,3.4); ax.axis("off")
box(ax,3.3,2.5,3.4,0.7,"CUDA Python",fc="#e8eef2",ec=INK,fs=15.6)
kids=[("cuda.core",LGREY,GREY),("cuda.bindings",LGREY,GREY),("cuda.compute",LGREEN,GREEN),("cuda.tile",LGREY,GREY),("nvmath-python",LGREY,GREY)]
w=1.75; gap=0.15; x0=(10-(len(kids)*w+(len(kids)-1)*gap))/2
for i,(nm,fc,ec) in enumerate(kids):
    x=x0+i*(w+gap); box(ax,x,0.7,w,0.8,nm,fc=fc,ec=ec,fs=12.6,bold=(nm=="cuda.compute"))
    arr(ax,(5,2.5),(x+w/2,1.5),c=GREY,lw=1.4)
ax.text(x0+2*(w+gap)+w/2,0.45,"↑ this talk",ha="center",fontsize=10.8,color=GREEN,fontweight="bold")
fig.savefig(f"{OUT}/ecosystem.png"); plt.close(fig)

# 1.3 abstraction spectrum
fig,ax=plt.subplots(figsize=(10,2.4)); ax.set_xlim(0,10); ax.set_ylim(0,2.4); ax.axis("off")
box(ax,0.2,0.9,3.0,1.0,"CuPy · PyTorch","array/tensor libraries",fc=LBLUE,ec=BLUE,fs=14.4)
box(ax,3.5,0.9,3.0,1.0,"cuda.compute","composable primitives",fc=LGREEN,ec=GREEN,fs=15.0)
box(ax,6.8,0.9,3.0,1.0,"CUDA C++","CUB · Thrust",fc=LGREY,ec=GREY,fs=14.4)
arr(ax,(3.25,1.4),(3.48,1.4)); arr(ax,(6.55,1.4),(6.78,1.4))
ax.text(0.2,0.45,"◀ higher level (applications)",fontsize=12.0,color="#5a6b70")
ax.text(9.8,0.45,"lower level (kernels) ▶",fontsize=12.0,color="#5a6b70",ha="right")
fig.savefig(f"{OUT}/spectrum.png"); plt.close(fig)

# 1.5 JIT pipeline
fig,ax=plt.subplots(figsize=(10,2.3)); ax.set_xlim(0,10); ax.set_ylim(0,2.3); ax.axis("off")
seq=[("Python operator",LGREEN,GREEN),("Numba CUDA",LGREY,GREY),("LTO-IR",LGREY,GREY),("JIT-link\n+ CUB/Thrust",LGREEN,GREEN),("one GPU kernel",LBLUE,BLUE)]
w=1.7; gap=0.35; x=0.15
for i,(nm,fc,ec) in enumerate(seq):
    box(ax,x,0.8,w,0.9,nm,fc=fc,ec=ec,fs=12.0)
    if i<len(seq)-1: arr(ax,(x+w,1.25),(x+w+gap,1.25))
    x+=w+gap
ax.text(5,0.4,"compiled once per (dtype, op) specialization — then cached",ha="center",fontsize=11.4,color="#5a6b70",style="italic")
fig.savefig(f"{OUT}/jit_pipeline.png"); plt.close(fig)

# 3.4 JIT vs AoT
fig,ax=plt.subplots(figsize=(10,2.6)); ax.set_xlim(0,10); ax.set_ylim(0,2.6); ax.axis("off")
ax.text(0.15,2.3,"JIT",fontsize=13.2,fontweight="bold",color=INK)
box(ax,1.2,1.8,2.4,0.7,"first call",sub="compile",fc=LRED,ec=RED,fs=12.0)
box(ax,4.0,1.8,2.4,0.7,"cached",sub="reused in session",fc=LGREY,ec=GREY,fs=12.0)
arr(ax,(3.6,2.15),(4.0,2.15))
ax.text(0.15,1.1,"AoT",fontsize=13.2,fontweight="bold",color=GREEN)
box(ax,1.2,0.55,2.4,0.7,"serialize",sub="compiled kernel → disk",fc=LGREEN,ec=GREEN,fs=12.0)
box(ax,4.0,0.55,2.4,0.7,"deserialize",sub="near-instant load",fc=LGREEN,ec=GREEN,fs=12.0)
box(ax,6.8,0.55,3.0,0.7,"no compile at run",fc=LBLUE,ec=BLUE,fs=12.0)
arr(ax,(3.6,0.9),(4.0,0.9),c=GREEN); arr(ax,(6.4,0.9),(6.8,0.9),c=GREEN)
ax.text(9.85,0.2,"ref: NVIDIA/cccl PR #9732",ha="right",fontsize=10.2,color="#7c8b90")
fig.savefig(f"{OUT}/aot.png"); plt.close(fig)

# 5.4 lazy array
fig,ax=plt.subplots(figsize=(10,2.7)); ax.set_xlim(0,10); ax.set_ylim(0,2.7); ax.axis("off")
ax.text(0.15,2.4,"today (eager)",fontsize=12.6,fontweight="bold",color=RED)
for i,op in enumerate(["op","op","op"]):
    x=1.4+i*2.3
    box(ax,x,1.7,1.4,0.6,op,fc=LGREY,ec=GREY,fs=12.0)
    box(ax,x,2.35,1.4,0.001,"",fc="none",ec="none")
    ax.text(x+0.7,1.5,"↓ kernel",ha="center",fontsize=9.6,color=RED)
ax.text(0.15,0.95,"future (lazy)",fontsize=12.6,fontweight="bold",color=GREEN)
box(ax,1.4,0.5,3.3,0.7,"expression graph",sub="capture whole chain",fc=LGREEN,ec=GREEN,fs=13.2)
box(ax,5.6,0.5,3.6,0.7,"one fused pipeline",sub="fuse ACROSS ops",fc=LBLUE,ec=BLUE,fs=13.2)
arr(ax,(4.7,0.85),(5.6,0.85),c=GREEN)
ax.text(9.85,0.15,"ref: scikit-hep/awkward #4173",ha="right",fontsize=10.2,color="#7c8b90")
fig.savefig(f"{OUT}/lazy_array.png"); plt.close(fig)
print("wrote ecosystem, spectrum, jit_pipeline, aot, lazy_array")
