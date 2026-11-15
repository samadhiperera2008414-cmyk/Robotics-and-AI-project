import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK="#0B1F2A"; TEAL="#2B7A78"; MINT="#DDEFEA"; AMBER="#F2A541"; GREY="#5B6B73"; WHITE="#FFFFFF"

def box(ax,x,y,w,h,text,fc,tc=WHITE,fs=9,ec=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec or fc,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold",linespacing=1.25)

def arrow(ax,x1,y1,x2,y2,c=GREY,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color=c,lw=1.5,linestyle=ls))

# 1. CNN architecture
fig,ax=plt.subplots(figsize=(12,3.3)); ax.set_xlim(0,24); ax.set_ylim(0,6.4); ax.axis("off")
blocks=[("Input\n32×32×3\n(raw 0–255)",INK),("Rescale\n÷255",GREY),
 ("Block 1\n2× Conv3×3(32)\nBN + ReLU\nMaxPool, Drop 0.2\n→ 16×16×32",TEAL),
 ("Block 2\n2× Conv3×3(64)\nBN + ReLU\nMaxPool, Drop 0.3\n→ 8×8×64",TEAL),
 ("Block 3\n2× Conv3×3(128)\nBN + ReLU\nMaxPool, Drop 0.4\n→ 4×4×128",TEAL),
 ("Flatten\n2048",GREY),("Dense 128\nBN + ReLU\nDrop 0.5",INK),("Dense 10\nSoftmax",AMBER)]
ws=[2.3,1.7,3.5,3.5,3.5,1.6,2.4,2.0]; x=0.2
for (t,c),w in zip(blocks,ws):
    box(ax,x,1.0,w,5.0,t,c,tc=INK if c==AMBER else WHITE,fs=11)
    if x>0.3: arrow(ax,x-0.28,3.5,x+0.02,3.5)
    x+=w+0.3
ax.text(12,0.3,"≈ 0.55 million parameters  |  Adam, cross-entropy loss, data augmentation",ha="center",fontsize=10.5,color=GREY)
fig.savefig("cnn_architecture.png",dpi=220,bbox_inches="tight",pad_inches=0.05,facecolor="white"); plt.close(fig)

# 2. Pipeline + offload
fig,ax=plt.subplots(figsize=(12,4.0)); ax.set_xlim(0,24); ax.set_ylim(0,8.0); ax.axis("off")
names=[("Camera\nframe",INK,WHITE),("HSV colour\ngate",TEAL,WHITE),("Crop to\n32×32",TEAL,WHITE),("Quantised\nCNN",AMBER,INK),("State\nmachine",INK,WHITE)]
xs=[0.2,5.0,9.8,14.6,19.4]; W=4.2
for (t,c,tc),x in zip(names,xs): box(ax,x,4.6,W,3.0,t,c,tc=tc,fs=14)
for i in range(4): arrow(ax,xs[i]+W+0.05,6.1,xs[i+1]-0.05,6.1)
box(ax,9.8,0.3,8.6,2.2,"Offload option: TCP inference server",WHITE,tc=INK,ec=TEAL,fs=13)
arrow(ax,11.9,4.55,11.9,2.55,c=TEAL,ls="--"); arrow(ax,16.4,2.55,21.5,4.55,c=TEAL,ls="--")
ax.text(0.2,1.4,"frame skipping\ncolour pre-filter\nresult cooldown",fontsize=11,color=GREY,va="center")
fig.savefig("pipeline.png",dpi=220,bbox_inches="tight",pad_inches=0.05,facecolor="white"); plt.close(fig)

# 3. FSM
fig,ax=plt.subplots(figsize=(12,4.2)); ax.set_xlim(0,27); ax.set_ylim(-1.2,6.4); ax.axis("off")
ks=["SEARCH","ALIGN","APPROACH","CLASSIFY","RECOVER"]
cols={"SEARCH":INK,"ALIGN":TEAL,"APPROACH":TEAL,"CLASSIFY":AMBER,"RECOVER":GREY}
xs=[0.3,5.9,11.5,17.1,22.7]; W=3.8
for k,x in zip(ks,xs): box(ax,x,2.2,W,1.8,k,cols[k],tc=INK if k=="CLASSIFY" else WHITE,fs=10)
lab=["colour blob\nconfirmed","object\ncentred","blob wide or\nwall close","CNN result\nlogged"]
for i in range(4):
    arrow(ax,xs[i]+W+0.05,3.1,xs[i+1]-0.05,3.1)
    ax.text((xs[i]+W+xs[i+1])/2,1.65,lab[i],ha="center",va="center",fontsize=8,color=GREY)
# blob lost -> SEARCH (dashed, above)
ax.add_patch(FancyArrowPatch((xs[2]+W/2,4.05),(xs[0]+W/2,4.05),connectionstyle="arc3,rad=0.28",arrowstyle="-|>",mutation_scale=10,color=GREY,lw=1.0,linestyle="--"))
ax.text(xs[1]+W/2+0.6,5.9,"blob lost > 8 frames → back to SEARCH",ha="center",fontsize=8,color=GREY)
# recover -> search (below)
ax.add_patch(FancyArrowPatch((xs[4]+W/2,2.15),(xs[0]+W/2,2.15),connectionstyle="arc3,rad=-0.22",arrowstyle="-|>",mutation_scale=12,color=GREY,lw=1.4))
ax.text(13.5,-0.35,"after reversing and turning away → resume SEARCH",ha="center",fontsize=8.5,color=GREY)
fig.savefig("fsm.png",dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("figs ok")
