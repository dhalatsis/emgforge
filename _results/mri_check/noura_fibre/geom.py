"""Shared geometry + the two incumbent (red/grey) morphing-disk methods,
   faithfully reimplemented from emgforge fiber_directions.py."""
import numpy as np, nibabel as nib
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

SEG="/home/noura/Documents/Projects/PhD/mri/data/Lab/WR/WR_Segmentation.nii.gz"

def load_mask(label):
    img=nib.load(SEG); d=img.get_fdata().astype(int); vs=np.array(img.header.get_zooms())
    return (d==label), vs, d

class Centerline:
    def __init__(s, mask, vs, smooth=1.0):
        vi=np.argwhere(mask)
        zs=np.unique(vi[:,2]); z=[];cx=[];cy=[]
        for zi in zs:
            sl=vi[vi[:,2]==zi]; ph=sl*vs
            z.append(ph[0,2]); cx.append(ph[:,0].mean()); cy.append(ph[:,1].mean())
        z=np.array(z);cx=np.array(cx);cy=np.array(cy)
        o=np.argsort(z); z,cx,cy=z[o],cx[o],cy[o]
        if smooth>0 and len(cx)>=3:
            cx=gaussian_filter1d(cx,smooth,mode="nearest"); cy=gaussian_filter1d(cy,smooth,mode="nearest")
        s.z0,s.z1=float(z[0]),float(z[-1])
        s.sx=CubicSpline(z,cx,bc_type="natural"); s.sy=CubicSpline(z,cy,bc_type="natural")
    def pos(s,z):
        z=np.clip(z,s.z0,s.z1); return np.stack([s.sx(z),s.sy(z),z],-1)

class CrossSection:
    def __init__(s, mask, vs, cl, ntheta=72, r_inset=0.92, smooth=1.5):
        s.theta=np.linspace(0,360,ntheta,endpoint=False); s.r_inset=r_inset
        step=min(vs[0],vs[1])*0.5; nx,ny=mask.shape[:2]
        zs=np.unique(np.argwhere(mask)[:,2]); Z=[];R=[]
        for zi in zs:
            sm=mask[:,:,zi]
            if sm.sum()<5: continue
            zmm=zi*vs[2]; p=cl.pos(np.array([zmm]))[0]; cx,cy=p[0],p[1]
            ci,cj=int(round(cx/vs[0])),int(round(cy/vs[1]))
            if not(0<=ci<nx and 0<=cj<ny and sm[ci,cj]):
                v2=np.argwhere(sm); cx=v2[:,0].mean()*vs[0]; cy=v2[:,1].mean()*vs[1]
            row=np.zeros(len(s.theta))
            for k,th in enumerate(np.radians(s.theta)):
                r=step
                while r<300:
                    ix=int(round((cx+r*np.cos(th))/vs[0])); iy=int(round((cy+r*np.sin(th))/vs[1]))
                    if not(0<=ix<nx and 0<=iy<ny) or not sm[ix,iy]: break
                    r+=step
                row[k]=max(r-step,step)
            row=gaussian_filter1d(row,smooth,mode="wrap")
            Z.append(zmm); R.append(row)
        s.Z=np.array(Z); s.R=np.array(R)  # (K,ntheta)
    def radius(s,z,theta_deg):
        # bilinear-ish: nearest z row, linear theta
        z=np.atleast_1d(z); out=np.zeros(len(z))
        for i,zz in enumerate(z):
            ki=np.argmin(np.abs(s.Z-np.clip(zz,s.Z[0],s.Z[-1])))
            t=theta_deg%360; 
            out[i]=np.interp(t, np.append(s.theta,360), np.append(s.R[ki],s.R[ki][0]))
        return out

def morphing_fiber(cl,cs,r_norm,theta_deg,zvals,taper=None):
    z=np.clip(zvals,cl.z0,cl.z1); cx=cl.sx(z); cy=cl.sy(z)
    R=cs.radius(z,theta_deg); scale=r_norm*cs.r_inset*R
    if taper is not None:
        zf=(z-cl.z0)/max(cl.z1-cl.z0,1e-9); scale=scale*(taper+(1-taper)*np.sin(np.pi*zf))
    th=np.radians(theta_deg)
    return np.stack([cx+scale*np.cos(th), cy+scale*np.sin(th), z],-1)

def pca_frame(mask,vs):
    c=np.argwhere(mask)*vs; cc=c-c.mean(0)
    w,V=np.linalg.eigh(np.cov(cc.T)); o=np.argsort(w)[::-1]
    return c.mean(0), V[:,o[0]], V[:,o[1]], V[:,o[2]]  # centroid, PC1(long),PC2,PC3(thin)

if __name__=="__main__":
    for L in (13,7):
        mask,vs,_=load_mask(L); cl=Centerline(mask,vs); cs=CrossSection(mask,vs,cl)
        ctr,p1,p2,p3=pca_frame(mask,vs)
        print(f"L{L}: zrange {cl.z0:.0f}-{cl.z1:.0f}mm ({cl.z1-cl.z0:.0f}mm), "
              f"{len(cs.Z)} xsec slices, Rmax {cs.R.max():.1f}mm, PC1={np.round(p1,2)}")
