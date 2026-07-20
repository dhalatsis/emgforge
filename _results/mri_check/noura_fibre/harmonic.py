"""Masked Laplace solve (insulated walls = zero-flux Neumann) + streamline tracer.
   This is the NEW method's engine: fibres = streamlines of a harmonic field whose
   BCs encode the aponeurosis topology. Contained + non-crossing by construction."""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label as cc_label
from geom import load_mask, pca_frame

def keep_largest(mask):
    lab,n=cc_label(mask)
    if n<=1: return mask
    sizes=np.bincount(lab.ravel()); sizes[0]=0
    return lab==sizes.argmax()

def solve_laplace(mask, vs, d0, d1):
    """phi harmonic, =0 on d0 voxels, =1 on d1, insulated (zero-flux) on muscle surface."""
    idx=np.full(mask.shape,-1,int); vox=np.argwhere(mask); 
    free=mask & ~d0 & ~d1
    fvox=np.argwhere(free); idx[free]=np.arange(len(fvox))
    N=len(fvox); A=lil_matrix((N,N)); b=np.zeros(N)
    h2=np.array([1/vs[0]**2,1/vs[1]**2,1/vs[2]**2])
    off=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    ax=[0,0,1,1,2,2]
    for n,(i,j,k) in enumerate(fvox):
        diag=0.0
        for (di,dj,dk),a in zip(off,ax):
            ii,jj,kk=i+di,j+dj,k+dk
            if not(0<=ii<mask.shape[0] and 0<=jj<mask.shape[1] and 0<=kk<mask.shape[2]): continue
            if not mask[ii,jj,kk]: continue   # outside muscle -> insulated (skip => zero flux)
            diag-=h2[a]
            if free[ii,jj,kk]: A[n,idx[ii,jj,kk]]+=h2[a]
            elif d1[ii,jj,kk]: b[n]-=h2[a]*1.0
            # d0 contributes 0
        A[n,n]=diag if diag!=0 else 1.0   # guard fully-insulated stray voxel
    phi=np.zeros(mask.shape); phi[d1]=1.0
    Acsr=csr_matrix(A) - 1e-9*identity(N)  # tiny Tikhonov vs residual singularity
    sol=spsolve(Acsr,b); phi[free]=sol
    return phi

def grad_field(phi,mask,vs):
    """Mask-aware gradient: at a wall voxel use a one-sided difference toward the
    in-mask neighbour, never against the phi=0 fill outside the muscle (which would
    corrupt the direction to ~normal-to-wall). Central diff in the interior."""
    g=np.zeros(phi.shape+(3,)); m=mask
    for a in range(3):
        h=vs[a]
        pp=np.roll(phi,-1,axis=a); pm=np.roll(phi,1,axis=a)   # phi[i+1], phi[i-1]
        mp=np.roll(m,-1,axis=a).copy(); mm=np.roll(m,1,axis=a).copy()
        last=[slice(None)]*3; last[a]=-1; mp[tuple(last)]=False   # kill roll wrap-around
        first=[slice(None)]*3; first[a]=0;  mm[tuple(first)]=False
        central=mp&mm
        g[...,a]=np.where(central,(pp-pm)/(2*h),
                  np.where(mp,(pp-phi)/h,
                   np.where(mm,(phi-pm)/h,0.0)))
    g[~m]=0
    return g

def trace(seed, g, mask, vs, step=1.0, maxlen=400, sign=1):
    """RK2 integrate normalized grad from seed (mm). Returns polyline (M,3)."""
    inv=1/vs; sh=np.array(mask.shape)
    def samp(p):
        # trilinear on grad
        c=p*inv; i0=np.floor(c).astype(int); f=c-i0
        acc=np.zeros(3); tot=0
        for dx in (0,1):
            for dy in (0,1):
                for dz in (0,1):
                    ii=i0+[dx,dy,dz]
                    if np.any(ii<0) or np.any(ii>=sh): continue
                    if not mask[ii[0],ii[1],ii[2]]: continue
                    w=(f[0] if dx else 1-f[0])*(f[1] if dy else 1-f[1])*(f[2] if dz else 1-f[2])
                    acc+=w*g[ii[0],ii[1],ii[2]]; tot+=w
        if tot<1e-9: return None
        v=acc/tot; n=np.linalg.norm(v)
        return v/n if n>1e-9 else None
    pts=[seed.copy()]; p=seed.copy(); L=0
    for _ in range(int(maxlen/step)):
        v=samp(p)
        if v is None: break
        v=v*sign
        pmid=p+0.5*step*v; vm=samp(pmid)
        if vm is None: break
        pn=p+step*(vm*sign)
        ci=np.round(pn*inv).astype(int)
        if np.any(ci<0) or np.any(ci>=sh) or not mask[ci[0],ci[1],ci[2]]: break
        pts.append(pn.copy()); L+=step; p=pn
    return np.array(pts)

def caps(mask,vs,frac=0.06):
    """proximal/distal cap voxel sets by z-extent."""
    z=np.argwhere(mask)[:,2]; z0,z1=z.min(),z.max(); w=max(int((z1-z0)*frac),1)
    d0=mask&(np.arange(mask.shape[2])[None,None,:]<=z0+w)
    d1=mask&(np.arange(mask.shape[2])[None,None,:]>=z1-w)
    return d0,d1

def surface(mask):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)

def aponeurosis_patches(mask,vs,ctr,p1,p3,ang_lon=(0.0,0.6),ang_lon2=(0.4,1.0)):
    """unipennate: patch0 on -p3 wall (proximal range), patch1 on +p3 wall (distal range)."""
    surf=surface(mask); sv=np.argwhere(surf)*vs
    s=(sv-ctr)@p1; s=(s-s.min())/(s.max()-s.min())     # longitudinal 0..1
    t3=(sv-ctr)@p3                                       # thin-axis signed
    svox=np.argwhere(surf)
    d0=np.zeros(mask.shape,bool); d1=np.zeros(mask.shape,bool)
    m0=(t3<0)&(s>=ang_lon[0])&(s<=ang_lon[1])
    m1=(t3>0)&(s>=ang_lon2[0])&(s<=ang_lon2[1])
    for v,fl in zip(svox,m0):
        if fl: d0[v[0],v[1],v[2]]=True
    for v,fl in zip(svox,m1):
        if fl: d1[v[0],v[1],v[2]]=True
    return d0,d1

if __name__=="__main__":
    L=13; mask,vs,_=load_mask(L); mask=keep_largest(mask); ctr,p1,p2,p3=pca_frame(mask,vs)
    d0,d1=caps(mask,vs); phi=solve_laplace(mask,vs,d0,d1); g=grad_field(phi,mask,vs)
    print("cap field solved. phi range",phi[mask].min(),phi[mask].max())
    # seed a few streamlines from mid cross-section
    mid=np.argwhere(mask); zc=int(np.median(mid[:,2]))
    sl=np.argwhere(mask[:,:,zc]); rng=np.random.default_rng(0)
    seeds=sl[rng.choice(len(sl),6,replace=False)]
    inside=0;tot=0
    for sxy in seeds:
        seed=np.array([sxy[0]*vs[0],sxy[1]*vs[1],zc*vs[2]])
        up=trace(seed,g,mask,vs,sign=1); dn=trace(seed,g,mask,vs,sign=-1)
        f=np.vstack([dn[::-1],up]); 
        # containment check
        ci=np.round(f/vs).astype(int); ok=mask[ci[:,0],ci[:,1],ci[:,2]].mean()
        print(f"  fiber len {len(f)}mm-steps, contained {ok*100:.0f}%")
