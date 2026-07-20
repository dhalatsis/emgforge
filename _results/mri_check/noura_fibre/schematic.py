import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(13,4.4))
for a in ax: a.set_xlim(-1.15,1.15); a.set_ylim(-1.12,1.12); a.set_aspect("equal"); a.axis("off")
W=0.62
def bound(y): return W*np.sqrt(np.clip(1-(np.abs(y)/1.02)**4,0,1))   # flat-ish spindle
def belly(a):
    y=np.linspace(-1.02,1.02,200); x=bound(y)
    a.fill_betweenx(y,-x,x,color="0.93",zorder=0); a.plot(x,y,"0.6",lw=1); a.plot(-x,y,"0.6",lw=1)
def fibre(a,x0,y0,x1,y1,color,lw=1.6,marks=True):
    # sample the straight fibre, clip inside the belly so nothing pokes out
    t=np.linspace(0,1,40); xs=x0+t*(x1-x0); ys=y0+t*(y1-y0); b=bound(ys)
    m=np.abs(xs)<=b+1e-3
    xs,ys=xs[m],ys[m]
    if len(xs)<2: return
    a.plot(xs,ys,color=color,lw=lw,zorder=2)
    if marks:
        a.scatter([xs[0]],[ys[0]],s=16,facecolors="none",edgecolors="#e08214",lw=1.0,zorder=5)   # begin
        a.scatter([xs[-1]],[ys[-1]],s=13,marker="s",facecolors="none",edgecolors="#d62728",lw=1.0,zorder=5)  # end
        i=len(xs)//2; a.scatter([xs[i]],[ys[i]],s=8,c="k",zorder=6)                                # NMJ / IZ

# 1. Fusiform / parallel
belly(ax[0])
for xo in np.linspace(-0.42,0.42,7): fibre(ax[0],xo,-0.95,xo,0.95,"#2ca02c")
ax[0].plot([0,0],[-1.05,-0.9],"k",lw=3); ax[0].plot([0,0],[0.9,1.05],"k",lw=3)
ax[0].plot([-0.28,0.28],[0,0],color="#8c2d04",lw=1.4,ls=":"); ax[0].text(0.46,0,"IZ",color="#8c2d04",fontsize=9,va="center")
ax[0].text(0,-1.09,"tendon",ha="center",fontsize=8)
ax[0].set_title("Fusiform / parallel\n(brachioradialis)\nlong fibres ≈ span belly, 1 IZ band",fontsize=10)

# 2. Unipennate — one aponeurosis, fibres tilt to it
belly(ax[1])
ax[1].plot([-0.42,-0.42],[-0.9,0.9],color="#1f77b4",lw=3)   # origin aponeurosis (left)
ax[1].plot([ 0.42, 0.42],[-0.9,0.9],color="#1f77b4",lw=3)   # insertion aponeurosis (right)
for y0 in np.linspace(-0.78,0.62,9): fibre(ax[1],-0.42,y0,0.42,y0+0.28,"#2ca02c")
ax[1].text(-0.42,1.0,"aponeurosis",fontsize=7,color="#1f77b4",ha="center")
ax[1].set_title("Unipennate\n(FPL, EPL)\nshort fibres tilt to ONE side",fontsize=10)

# 3. Bipennate — central tendon, two sets converge on it
belly(ax[2])
ax[2].plot([0,0],[-0.95,0.95],color="#d62728",lw=3)          # central tendon
ax[2].plot([-0.42,-0.42],[-0.9,0.9],color="#1f77b4",lw=2.3)  # outer aponeuroses
ax[2].plot([ 0.42, 0.42],[-0.9,0.9],color="#1f77b4",lw=2.3)
for y0 in np.linspace(-0.78,0.6,8):
    fibre(ax[2],-0.42,y0,0,y0+0.20,"#1a9850")               # left set -> central tendon
    fibre(ax[2], 0.42,y0,0,y0+0.20,"#762a83")               # right set -> central tendon
ax[2].text(0,-1.06,"central tendon",ha="center",fontsize=8,color="#d62728")
ax[2].text(-0.62,0.95,"outer\naponeurosis",fontsize=7,color="#1f77b4",ha="center")
ax[2].set_title("Bipennate (herringbone)\n(FCU, ECU, pronator teres)\ntwo sets → central tendon",fontsize=10)

ax[0].scatter([],[],s=16,facecolors="none",edgecolors="#e08214",lw=1.0,label="fibre begin")
ax[0].scatter([],[],s=13,marker="s",facecolors="none",edgecolors="#d62728",lw=1.0,label="fibre end")
ax[0].scatter([],[],s=10,c="k",label="innervation zone (NMJ)")
fig.legend(*ax[0].get_legend_handles_labels(),loc="lower center",ncol=3,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.suptitle("Muscle fibre architecture — how fibres attach, and why they don't span the muscle",fontsize=12,y=1.03)
plt.tight_layout(); plt.savefig("schematic.png",dpi=115,bbox_inches="tight"); print("saved")
