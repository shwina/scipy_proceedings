import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; GREEN="#4c8c00"; LGREEN="#e4f2d0"; BLUE="#2b6cb0"; LBLUE="#dbe7f4"
GREY="#9aa8ad"; PURP="#7c3aed"; LPURP="#ece3f5"
OUT="figs"

def box(ax,x,y,w,h,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",fc=fc,ec=ec,lw=2))
def mono(ax,x,y,txt,fs=11.4,color="#2c3b41",ha="left"):
    ax.text(x,y,txt,ha=ha,va="top",fontsize=fs,family="monospace",color=color)
def arr(ax,p1,p2,c=INK,lw=2.0):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=15,lw=lw,color=c))

fig,ax=plt.subplots(figsize=(10.6,7.0)); ax.set_xlim(0,14); ax.set_ylim(0,9.6); ax.axis("off")

# ---- top: the Python call ----
box(ax,2.6,8.05,8.8,1.30,LBLUE,BLUE)
ax.text(7.0,9.05,"Python call",ha="center",va="center",fontsize=12.6,fontweight="bold",color=BLUE)
mono(ax,3.0,8.66,"unary_transform(d_in=x, d_out=y,",fs=11.4)
mono(ax,3.0,8.36,"                op=square, num_items=n)",fs=11.4)

# split point
arr(ax,(7.0,8.05),(3.4,6.7),c=GREY)   # to left
arr(ax,(7.0,8.05),(10.6,6.7),c=GREY)  # to right

# ---- left: CUDA C++ kernel (CUB) -> NVRTC ----
box(ax,0.3,3.5,6.3,3.1,LGREEN,GREEN)
ax.text(3.45,6.28,"CUDA C++ Kernel (CUB)",ha="center",va="center",fontsize=13.8,fontweight="bold",color=INK)
mono(ax,0.6,5.75,"cub::...::transform_kernel<",fs=10.8)
mono(ax,0.6,5.45,"    device_transform_policy,",fs=10.8)
mono(ax,0.6,5.15,"    T,",fs=10.8)
mono(ax,0.6,4.85,"    cccl_op{ .operation = my_op },",fs=10.8)
mono(ax,0.6,4.55,"    ... >",fs=10.8)
ax.text(3.45,3.95,"compiled with NVRTC   (C++ → LTO-IR)",ha="center",fontsize=11.4,style="italic",color=GREEN)

# ---- right: Python operator -> Numba ----
box(ax,7.4,3.5,6.3,3.1,LPURP,PURP)
ax.text(10.55,6.28,"Python operator",ha="center",va="center",fontsize=13.8,fontweight="bold",color=INK)
mono(ax,7.7,5.75,"def square(v):",fs=11.4)
mono(ax,7.7,5.45,"    return v * v",fs=11.4)
mono(ax,7.7,4.85,'numba.cuda.compile(square,',fs=11.4)
mono(ax,7.7,4.55,'                   output="ltoir")',fs=11.4)
ax.text(10.55,3.95,"compiled with Numba   (Python → LTO-IR)",ha="center",fontsize=11.4,style="italic",color=PURP)

# converge
arr(ax,(3.45,3.5),(6.4,1.95),c=GREEN)
arr(ax,(10.55,3.5),(7.6,1.95),c=PURP)

# ---- bottom: nvJitLink -> linked CUDA kernel ----
box(ax,4.3,0.55,5.4,1.35,LGREEN,GREEN)
ax.text(7.0,1.52,"nvJitLink",ha="center",va="center",fontsize=15.6,fontweight="bold",color=INK,family="monospace")
ax.text(7.0,1.02,"linked CUDA kernel",ha="center",va="center",fontsize=12.6,color="#3a4a50")

fig.savefig(f"{OUT}/jit_compile.png"); plt.close(fig)
print("wrote figs/jit_compile.png")
