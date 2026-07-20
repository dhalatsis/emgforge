import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import densefib as DF

def mark_ends(ax,f,proj):
    u,w=proj(f)
    ax.scatter([u[0]],[w[0]],s=20,facecolors="none",edgecolors="#e08214",lw=1.0,zorder=6)       # begin
    ax.scatter([u[-1]],[w[-1]],s=16,marker="s",facecolors="none",edgecolors="#d62728",lw=1.0,zorder=6)  # end
def mark_nmj(ax,pt,proj,is_atlas):
    mu,mw=proj(pt)
    if is_atlas: ax.scatter([mu],[mw],s=13,c="k",zorder=7)                                        # measured IZ
    else: ax.scatter([mu],[mw],s=11,facecolors="none",edgecolors="0.45",lw=0.9,zorder=7)          # inferred fill

def compare_fig(L, out, title):
    D=DF.setup(L); red=DF.morph(D,None); grey=DF.morph(D,0.35)
    cols=DF.sparse_columns_iz(D,ncol=8)                    # sparse in-series stacks, IZ-pinned
    dense,dband,dnmj,disatl=DF.dense_fill_iz(D,grid_mm=2.0)# full-density fill (its own row)
    gap,ns=DF.min_gap(D,grid=4.0)
    atlas=DF._iz_params(D)[0]
    ctr,p1,p3=D["ctr"],D["p1"],D["p3"]
    def proj(P): d=np.atleast_2d(P)-ctr; return d@p1,d@p3
    sv=np.argwhere(DF.surface(D["mask"]))*D["vs"]; U,W=proj(sv)
    fig,ax=plt.subplots(4,1,figsize=(11,10.6))
    rows=[("morph","#d62728","RED — morphing disk: one fibre spans the whole muscle, single NMJ at mid-length",red),
          ("morph","#7f7f7f","GREY — fusiform taper: same, one full-length fibre, single mid NMJ",grey),
          ("sparse",None,f"NEW (representative) — in-series fibres (Lf={D['Lf']:.0f} mm); NMJs PINNED to atlas IZ {atlas} (black) + inferred fill (grey)",cols),
          ("dense",None,f"NEW (fully filled) — the whole belly, {len(dense)} fibres; same begin/end/IZ; the black NMJs line up into the atlas bands",None)]
    for a,(kind,col,ttl,data) in zip(ax,rows):
        a.scatter(U,W,s=2,c="0.9",alpha=.45,zorder=0)
        if kind=="morph":
            for f in data:
                u,w=proj(f); a.plot(u,w,color=col,lw=1.1,alpha=.9,zorder=2); mark_ends(a,f,proj)
                arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))])
                mark_nmj(a,f[np.argmin(np.abs(arc-arc[-1]/2))],proj,True)   # their lone mid-muscle NMJ
        elif kind=="sparse":
            for stack in data:
                for seg,bd,nm,atl in stack:
                    u,w=proj(seg); a.plot(u,w,color=DF.BANDCOL[bd%len(DF.BANDCOL)],lw=1.2,alpha=.9,zorder=2)
                    mark_ends(a,seg,proj); mark_nmj(a,nm,proj,atl)
            a.text(0.01,0.06,f"fibres never cross: {ns} streamlines, min gap {gap:.1f} mm",
                   transform=a.transAxes,fontsize=7.5,color="0.3",style="italic")
        else:  # dense
            for seg,bd in zip(dense,dband):
                u,w=proj(seg); a.plot(u,w,color=DF.BANDCOL[bd%len(DF.BANDCOL)],lw=0.5,alpha=.7,zorder=2)
            for seg in dense:
                u,w=proj(seg)
                a.scatter([u[0]],[w[0]],s=5,facecolors="none",edgecolors="#e08214",lw=0.5,alpha=.7,zorder=5)
                a.scatter([u[-1]],[w[-1]],s=4,marker="s",facecolors="none",edgecolors="#d62728",lw=0.5,alpha=.7,zorder=5)
            if len(dnmj):
                nA=dnmj[disatl]; nF=dnmj[~disatl]
                if len(nA): uu,ww=proj(nA); a.scatter(uu,ww,s=6,c="k",zorder=6)
                if len(nF): uu,ww=proj(nF); a.scatter(uu,ww,s=5,facecolors="none",edgecolors="0.45",lw=0.6,zorder=6)
        a.set_aspect("equal"); a.set_title(ttl,fontsize=8.8,loc="left"); a.set_ylabel("across [mm]")
    # legend on the top row
    ax[0].scatter([],[],s=20,facecolors="none",edgecolors="#e08214",lw=1.0,label="fibre begin")
    ax[0].scatter([],[],s=16,marker="s",facecolors="none",edgecolors="#d62728",lw=1.0,label="fibre end")
    ax[0].scatter([],[],s=13,c="k",label="IZ pinned to atlas (measured)")
    ax[0].scatter([],[],s=11,facecolors="none",edgecolors="0.45",lw=0.9,label="IZ inferred (fill)")
    ax[0].legend(fontsize=7,loc="upper right",ncol=2,frameon=True)
    ax[-1].set_xlabel("along muscle (length) [mm]")
    fig.suptitle(title,fontsize=12); plt.tight_layout(); plt.savefig(out,dpi=104,bbox_inches="tight"); plt.close()
    print(out,"->",sum(len(c) for c in cols),"sparse,",len(dense),"dense, min gap",round(gap,2))

compare_fig(13,"compare_methods.png","FDS (label 13) — three fibre-geometry methods, with fibre begin / end / innervation zone")
compare_fig(8,"fcu_compare.png","FCU (label 8) — three fibre-geometry methods, with fibre begin / end / innervation zone")

# showcase: 3D dense fill (coarser) + non-crossing, with NMJ dots in 3D
def showcase_fig(L,out):
    D=DF.setup(L); segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=4.0)
    mask,vs,ctr,p1,p2,p3=D["mask"],D["vs"],D["ctr"],D["p1"],D["p2"],D["p3"]
    sv=np.argwhere(DF.surface(mask))*vs
    fig=plt.figure(figsize=(14,5.4))
    ax1=fig.add_subplot(121,projection="3d")
    ax1.scatter(sv[::6,0],sv[::6,1],sv[::6,2],s=1,c="0.8",alpha=0.08)
    for s,bd in zip(segs,band): ax1.plot(s[:,0],s[:,1],s[:,2],color=DF.BANDCOL[bd%len(DF.BANDCOL)],lw=0.8,alpha=.8)
    nmj=np.array(nmj)
    if isatl.any(): ax1.scatter(*nmj[isatl].T,s=9,c="k",depthshade=False,label="IZ (atlas)")       # measured
    if (~isatl).any(): ax1.scatter(*nmj[~isatl].T,s=7,facecolors="none",edgecolors="0.45",depthshade=False,label="IZ (fill)")  # inferred
    ax1.set_box_aspect((np.ptp(sv[:,0]),np.ptp(sv[:,1]),np.ptp(sv[:,2])))
    ax1.set_title(f"NEW — {D['key']} in 3D ({len(segs)} fibres; NMJs pinned to atlas IZ)",fontsize=9)
    ax1.set_xlabel("x",fontsize=7);ax1.set_ylabel("y",fontsize=7);ax1.set_zlabel("z",fontsize=7)
    ax1.view_init(elev=14,azim=-70)
    for ax in (ax1.xaxis,ax1.yaxis,ax1.zaxis): ax.set_major_locator(plt.MaxNLocator(4))
    ax1.tick_params(labelsize=6)
    from scipy.spatial.distance import pdist
    ax2=fig.add_subplot(122)
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])*vs[:2]
    zg=np.linspace(D["zlo"]+5,D["zhi"]-5,40); R=[]
    def stream(seed):
        up=DF.trace(seed,D["g"],mask,vs,sign=1,step=1.0,maxlen=320); dn=DF.trace(seed,D["g"],mask,vs,sign=-1,step=1.0,maxlen=320)
        f=np.vstack([dn[::-1],up]); return f[(f[:,2]>=D["zlo"])&(f[:,2]<=D["zhi"])]
    for gx in np.arange(sm[:,0].min(),sm[:,0].max(),3.5):
        for gy in np.arange(sm[:,1].min(),sm[:,1].max(),3.5):
            c=np.array([int(round(gx/vs[0])),int(round(gy/vs[1])),zi])
            if not(0<=c[0]<mask.shape[0] and 0<=c[1]<mask.shape[1] and mask[tuple(c)]): continue
            sf=stream(np.array([gx,gy,zc]))
            if len(sf)<15: continue
            r=np.stack([np.interp(zg,sf[:,2],sf[:,0]),np.interp(zg,sf[:,2],sf[:,1])],1)
            if all(np.min(np.linalg.norm(rk-r,axis=1))>1.5 for rk in R): R.append(r)
    R=np.array(R); gmin=min(pdist(R[:,z,:]).min() for z in range(len(zg))) if len(R)>1 else 0
    ax2.scatter(sm[:,0],sm[:,1],s=6,c="0.85")
    mids=R[:,len(zg)//2,:]; ax2.scatter(mids[:,0],mids[:,1],s=30,c=plt.cm.viridis(np.linspace(0,1,len(mids))),edgecolor="k",lw=.3)
    ax2.set_aspect("equal"); ax2.set_title(f"Non-crossing: {len(R)} streamlines, min gap {gmin:.1f} mm",fontsize=9)
    ax2.set_xlabel("x (mm)");ax2.set_ylabel("y (mm)")
    fig.suptitle(f"{D['key']} — filled in 3D (NMJ marked), and fibres never cross",fontsize=12)
    plt.tight_layout(); plt.savefig(out,dpi=105,bbox_inches="tight"); plt.close(); print(out,"->",len(segs),"fibres, min gap",round(gmin,2))
showcase_fig(8,"showcase.png")
