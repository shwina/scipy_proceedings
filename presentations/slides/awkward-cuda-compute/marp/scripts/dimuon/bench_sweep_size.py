import sys, time, statistics as st
sys.path.insert(0, "/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/scripts/dimuon")
import awkward as ak, cupy as cp, numpy as np, uproot
from cuda.compute import PermutationIterator, ZipIterator, gpu_struct, binary_transform
from ak_helpers import get_example2_buffers, make_like_offsets

file = uproot.open("https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.root")
arrays = file["Events"].arrays(filter_name="/Muon_(pt|eta|phi|charge)/")
base = ak.zip({"pt":arrays["Muon_pt"],"eta":arrays["Muon_eta"],"phi":arrays["Muon_phi"],"charge":arrays["Muon_charge"]})
print("base events:", len(base))

def before_mass(mu1, mu2):
    return np.sqrt(2*mu1.pt*mu2.pt*(np.cosh(mu1.eta-mu2.eta)-np.cos(mu1.phi-mu2.phi)))
@gpu_struct
class Muon:
    pt: np.float32; eta: np.float32; phi: np.float32; charge: np.int32
def op(m1,m2):
    return (2*m1[0]*m2[0]*(np.cosh(m1[1]-m2[1])-np.cos(m1[2]-m2[2])))**0.5
def after_mass(mu1, mu2, out):
    o1,i1,pt1,e1,p1,c1 = get_example2_buffers(mu1)
    o2,i2,pt2,e2,p2,c2 = get_example2_buffers(mu2)
    d1=PermutationIterator(ZipIterator(pt1,e1,p1,c1),i1); d2=PermutationIterator(ZipIterator(pt2,e2,p2,c2),i2)
    binary_transform(d_in1=d1,d_in2=d2,d_out=out.layout.content.data,op=op,num_items=len(i1))
def bench(fn,w=3,r=8):
    for _ in range(w): fn()
    cp.cuda.Device().synchronize(); ts=[]
    for _ in range(r):
        cp.cuda.Device().synchronize(); t=time.perf_counter(); fn(); cp.cuda.Device().synchronize()
        ts.append((time.perf_counter()-t)*1e3)
    return st.median(ts)

print(f"{'K':>6}{'pairs':>14}{'~read MB':>10}{'before ms':>11}{'after ms':>10}{'speedup':>9}   (L2=96MB)")
for K in [100, 400]:
    muons = ak.concatenate([base]*K)
    pairs = ak.to_backend(ak.combinations(muons, 2), "cuda")
    mu1, mu2 = ak.unzip(pairs)
    npair = int(ak.count(mu1.pt))
    mb = npair*6*4/1e6
    out = make_like_offsets(mu1.pt)
    tb = bench(lambda: before_mass(mu1,mu2)); ta = bench(lambda: after_mass(mu1,mu2,out))
    print(f"{K:>6}{npair:>14,}{mb:>10.1f}{tb:>11.2f}{ta:>10.2f}{tb/ta:>8.2f}x")
    del muons, pairs, mu1, mu2, out; cp.get_default_memory_pool().free_all_blocks()
