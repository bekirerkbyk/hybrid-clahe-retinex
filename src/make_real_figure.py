import os, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import methods as M
plt.rcParams.update({"font.size":10,"savefig.dpi":200,"savefig.bbox":"tight"})

REAL=os.path.join(os.path.dirname(__file__),"..","data","real")
FIG=os.path.join(os.path.dirname(__file__),"..","figures")
imgs=["DICM_01.jpg","DICM_05.jpg","LIME_2.bmp"]  # 3 representative real low-light
def load(p):
    bgr=cv2.imread(os.path.join(REAL,p))
    h,w=bgr.shape[:2]
    if max(h,w)>512:
        s=512/max(h,w); bgr=cv2.resize(bgr,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    return bgr
def b2r(x): return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

rows=len(imgs); cols=4
fig,axes=plt.subplots(rows,cols,figsize=(13,3.0*rows),
                      gridspec_kw={"hspace":0.22,"wspace":0.06})
titles=["Real low-light input","GHE","SSR-only","Proposed (adaptive)"]
for i,name in enumerate(imgs):
    img=load(name)
    outs=[img, M.METHODS["GHE"](img), M.METHODS["SSR-only"](img,sigma=30),
          M.METHODS["Proposed"](img,clip_limit=4,sigma=30,k=0.35)]
    for j,(o,t) in enumerate(zip(outs,titles)):
        ax=axes[i,j] if rows>1 else axes[j]
        ax.imshow(b2r(o)); ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(t,fontsize=10)
        if j==0: ax.set_ylabel(name.split('.')[0],fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#bbb")
fig.suptitle("Qualitative results on REAL low-light images (DICM / LIME) — no synthetic degradation",fontsize=11,y=0.995)
fig.savefig(os.path.join(FIG,"fig_real.png"))
print("saved fig_real.png")
