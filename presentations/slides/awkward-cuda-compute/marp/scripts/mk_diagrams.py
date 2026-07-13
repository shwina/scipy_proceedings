import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
OUT="figs"
INK="#17303a"; GREY="#8a969b"
SEQ=["#2b6cb0","#4c8c00","#b5651d","#7c3aed","#b23b2e","#0e7c7b","#c026a6","#846b00"]
def sq(ax,x,y,s,fc,txt=None,tc="white",fs=14.4,ec="white"):
    ax.add_patch(Rectangle((x,y),s,s,fc=fc,ec=ec,lw=1.6))
    if txt is not None: ax.text(x+s/2,y+s/2,txt,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold")
def arr(ax,p1,p2,c=INK,lw=1.6):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=11,lw=lw,color=c,shrinkA=1,shrinkB=1))

# ============ algorithms.png : 2x2 panels ============
fig,axs=plt.subplots(2,2,figsize=(9.0,6.2))
for a in axs.flat: a.set_xlim(0,10); a.set_ylim(0,7); a.axis("off")

# --- reduce ---
ax=axs[0,0]; ax.set_title("reduce", fontsize=16.8, fontweight="bold", color=INK)
n=4; s=1.3; x0=(10-n*(s+0.25))/2
for i in range(n): sq(ax,x0+i*(s+0.25),5.2,s,SEQ[i],"")
# tree to one
mid1=[x0+0*(s+0.25)+ (s+0.25)*0.5, x0+2*(s+0.25)+(s+0.25)*0.5]
for i in range(n): arr(ax,(x0+i*(s+0.25)+s/2,5.2),(5-0.65+ (0 if i<2 else 1.3),3.3),c=GREY)
sq(ax,5-0.65,2.0,s,INK,"Σ",fs=18.0);

# --- scan ---
ax=axs[0,1]; ax.set_title("scan", fontsize=16.8, fontweight="bold", color=INK)
vals=[3,1,4,2]; run=[]
acc=0
for v in vals: acc+=v; run.append(acc)
n=4; x0=(10-n*(s+0.25))/2
for i in range(n): sq(ax,x0+i*(s+0.25),5.0,s,SEQ[i],str(vals[i]))
for i in range(n):
    sq(ax,x0+i*(s+0.25),2.1,s,SEQ[i],str(run[i]))
    arr(ax,(x0+i*(s+0.25)+s/2,5.0),(x0+i*(s+0.25)+s/2,3.4),c=GREY)
    if i>0: arr(ax,(x0+(i-1)*(s+0.25)+s,2.1+s/2),(x0+i*(s+0.25),2.1+s/2),c=INK)

# --- transform ---
ax=axs[1,0]; ax.set_title("transform", fontsize=16.8, fontweight="bold", color=INK)
n=4; x0=(10-n*(s+0.25))/2
for i in range(n): sq(ax,x0+i*(s+0.25),5.0,s,SEQ[i],"")
for i in range(n):
    sq(ax,x0+i*(s+0.25),2.1,s,SEQ[i],"",ec=INK)
    arr(ax,(x0+i*(s+0.25)+s/2,5.0),(x0+i*(s+0.25)+s/2,3.4),c=INK)
ax.text(5,3.75,"apply f",ha="center",fontsize=12.0,color=INK,style="italic")

# --- segmented_reduce ---
ax=axs[1,1]; ax.set_title("segmented reduce", fontsize=16.8, fontweight="bold", color=INK)
segs=[3,2,3]; cols=[SEQ[0],SEQ[1],SEQ[2]]; x=0.4; tops=[]
for gi,gn in enumerate(segs):
    cx=[]
    for k in range(gn):
        sq(ax,x,5.0,1.05,cols[gi],""); cx.append(x+1.05/2); x+=1.15
    tops.append(cx); x+=0.5
outx=[]
for gi,cx in enumerate(tops):
    ox=sum(cx)/len(cx)
    sq(ax,ox-0.55,2.0,1.1,cols[gi],"Σ",fs=15.6); outx.append(ox)
    for c in cx: arr(ax,(c,5.0),(ox,3.2),c=GREY)

fig.suptitle("Algorithms: composable parallel building blocks", fontsize=18.0, fontweight="bold", color="#123f4d", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/algorithms.png"); plt.close(fig)
print("wrote algorithms.png")

# ============ iterators_seq.png : 4 rows, real sequences ============
fig,ax=plt.subplots(figsize=(9.0,7.2)); ax.set_xlim(0,12); ax.set_ylim(0,14); ax.axis("off")
def cell(x,y,w,txt,fc="#eef3f5",ec="#9aa8ad",tc=INK,fs=15.6,bold=False):
    ax.add_patch(Rectangle((x,y),w,0.95,fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+0.48,txt,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold" if bold else "normal")

# CountingIterator
ax.text(0.2,13.2,"CountingIterator(0)",fontsize=15.6,fontweight="bold",color="#123f4d")
xs=0.4
for v in ["0","1","2","3","4","…"]:
    cell(xs,12.0,1.1,v); xs+=1.25

# TransformIterator
ax.text(0.2,10.7,"TransformIterator(x, lambda i: i*i)",fontsize=15.6,fontweight="bold",color="#123f4d")
xs=0.4
for v in ["0","1","2","3","4"]:
    cell(xs,9.5,1.1,v); xs+=1.25
xs=0.4
for v in ["0","1","4","9","16"]:
    cell(xs,8.1,1.1,v,fc="#e4f2d0",ec="#4c8c00")
    arr(ax,(xs+0.55,9.5),(xs+0.55,9.05),c="#4c8c00")
    xs+=1.25

# PermutationIterator (a lazy gather)
ax.text(0.2,6.7,"PermutationIterator(vals, order)   order = [2, 0, 3]",fontsize=15.6,fontweight="bold",color="#123f4d")
pv=["10","20","30","40"]; vx=[]
xs=0.4
for v in pv:
    cell(xs,5.5,1.1,v,fc="#dbe7f4",ec="#2b6cb0"); vx.append(xs+0.55); xs+=1.25
xs=0.4
for v,src in [("30",2),("10",0),("40",3)]:
    cell(xs,4.1,1.1,v,fc="#e4f2d0",ec="#4c8c00")
    arr(ax,(vx[src],5.5),(xs+0.55,5.05),c="#4c8c00")
    xs+=1.25

# ZipIterator
ax.text(0.2,2.7,"ZipIterator(vals, index)",fontsize=15.6,fontweight="bold",color="#123f4d")
vals=["3.1","1.4","2.7"]; idx=["0","1","2"]
xs=0.4
for i in range(3):
    cell(xs,1.3,1.0,vals[i],fc="#dbe7f4",ec="#2b6cb0")
    cell(xs+1.05,1.3,0.9,idx[i],fc="#f0e6f8",ec="#7c3aed")
    ax.text(xs+1.0,0.85,f"({vals[i]}, {idx[i]})",ha="center",fontsize=11.4,color=GREY)
    xs+=2.6
fig.savefig(f"{OUT}/iterators_seq.png"); plt.close(fig)
print("wrote iterators_seq.png")
