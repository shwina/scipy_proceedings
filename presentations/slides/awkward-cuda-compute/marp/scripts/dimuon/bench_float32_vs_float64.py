import sys,time,math,statistics as st; sys.path.insert(0,"/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/scripts/dimuon")
import awkward as ak, cupy as cp, numpy as np, uproot
from cuda.compute import PermutationIterator, ZipIterator, gpu_struct, binary_transform
from ak_helpers import get_example2_buffers, make_like_offsets
arrays=uproot.open("https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.root")["Events"].arrays(filter_name="/Muon_(pt|eta|phi|charge)/")
base=ak.zip({"pt":arrays["Muon_pt"],"eta":arrays["Muon_eta"],"phi":arrays["Muon_phi"],"charge":arrays["Muon_charge"]})
mu1,mu2=ak.unzip(ak.to_backend(ak.combinations(ak.concatenate([base]*100),2),"cuda"))
npair=int(ak.count(mu1.pt)); print("pairs:",npair)
@gpu_struct
class Muon:
    pt: np.float32; eta: np.float32; phi: np.float32; charge: np.int32
def op64(m1,m2):  # original: promotes to float64
    return (2*m1[0]*m2[0]*(np.cosh(m1[1]-m2[1])-np.cos(m1[2]-m2[2])))**0.5
def op32(m1,m2):  # force float32 throughout
    f=np.float32
    de=m1[1]-m2[1]; dp=m1[2]-m2[2]
    v=f(2.0)*m1[0]*m2[0]*(f(math.cosh(de))-f(math.cos(dp)))
    return f(math.sqrt(f(v)))
o1,i1,a1,b1,c1,d1_=get_example2_buffers(mu1); o2,i2,a2,b2,c2,d2_=get_example2_buffers(mu2)
D1=PermutationIterator(ZipIterator(a1,b1,c1,d1_),i1); D2=PermutationIterator(ZipIterator(a2,b2,c2,d2_),i2)
out=make_like_offsets(mu1.pt); dout=out.layout.content.data
def run(op): binary_transform(d_in1=D1,d_in2=D2,d_out=dout,op=op,num_items=len(i1))
def bench(op,w=3,r=12):
    for _ in range(w): run(op)
    cp.cuda.Device().synchronize(); ts=[]
    for _ in range(r):
        cp.cuda.Device().synchronize(); t=time.perf_counter(); run(op); cp.cuda.Device().synchronize(); ts.append((time.perf_counter()-t)*1e3)
    return st.median(ts)
def before(): return np.sqrt(2*mu1.pt*mu2.pt*(np.cosh(mu1.eta-mu2.eta)-np.cos(mu1.phi-mu2.phi)))
def benchb(w=3,r=12):
    for _ in range(w): before()
    cp.cuda.Device().synchronize(); ts=[]
    for _ in range(r):
        cp.cuda.Device().synchronize(); t=time.perf_counter(); before(); cp.cuda.Device().synchronize(); ts.append((time.perf_counter()-t)*1e3)
    return st.median(ts)
tb=benchb(); t64=bench(op64); t32=bench(op32)
print(f"before (cupy ufuncs, float32) : {tb:7.2f} ms")
print(f"after  binary_transform float64: {t64:7.2f} ms   speedup {tb/t64:.2f}x")
print(f"after  binary_transform float32: {t32:7.2f} ms   speedup {tb/t32:.2f}x")
