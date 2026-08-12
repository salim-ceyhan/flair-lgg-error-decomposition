"""Experimental FACSeg-Fast operator with an image-adaptive spatial beta field.

This module is an ablation and does not replace the canonical implementation.
The beta field is frozen from the input image. Spatial derivatives of B=beta^2
are included in the historical FACSeg-Fast metric-derivative approximation.
"""
from __future__ import annotations
import numpy as np

def _derivatives(img):
    m,n=img.shape; cf=np.clip(np.arange(n)+1,0,n-1); cb=np.clip(np.arange(n)-1,0,n-1)
    rf=np.clip(np.arange(m)+1,0,m-1); rb=np.clip(np.arange(m)-1,0,m-1)
    ix=(img[:,cf]-img[:,cb])/2; iy=(img[rf,:]-img[rb,:])/2
    ixx=img[:,cf]-2*img+img[:,cb]; iyy=img[rf,:]-2*img+img[rb,:]
    ixy=(img[np.ix_(rf,cf)]+img[np.ix_(rb,cb)]-img[np.ix_(rb,cf)]-img[np.ix_(rf,cb)])/4
    return ix,iy,ixx,iyy,ixy

def _first(field):
    ix,iy,*_=_derivatives(field); return ix,iy

def _direction(image):
    diffs=np.stack([np.abs(image-np.roll(np.roll(image,dr-1,0),dc-1,1))
                    for dr in range(3) for dc in range(3)])
    k=np.argmax(diffs,axis=0); return (k//3).astype(float)-1,(k%3).astype(float)-1

def spatial_beta_field(image: np.ndarray, beta0: float=5.0, percentile: float=95.0,
                       lower_ratio: float=.5, upper_ratio: float=1.5) -> np.ndarray:
    """Robust edge-equalising beta: beta0 at P95, larger below and smaller above."""
    image=np.asarray(image,float); ix,iy,*_=_derivatives(image); gm=np.hypot(ix,iy)
    foreground=image>0.05; values=gm[foreground] if foreground.any() else gm.ravel()
    scale=float(np.percentile(values,percentile))+1e-12
    ratio=np.clip(2*scale/(scale+gm),lower_ratio,upper_ratio)
    return float(beta0)*ratio

def local_reliability_beta_field(image: np.ndarray, beta0: float=5.0,
                                 percentile: float=95.0) -> np.ndarray:
    """Restrict edge equalisation to the established FLAIR evidence window.

    The soft window is near one for plausible lesion intensities and near zero
    elsewhere. Therefore beta returns continuously to beta0 outside the local
    evidence support rather than altering the geometry of the entire brain.
    """
    image=np.asarray(image,float); ix,iy,*_=_derivatives(image); gm=np.hypot(ix,iy)
    foreground=image>0.05; values=gm[foreground] if foreground.any() else gm.ravel()
    scale=float(np.percentile(values,percentile))+1e-12
    softness=.05
    window=(1/(1+np.exp(-(image-.28)/softness)))*(1/(1+np.exp((image-.82)/softness)))
    equalising_ratio=np.clip(2*scale/(scale+gm),.5,1.5)
    ratio=1+window*(equalising_ratio-1)
    return float(beta0)*ratio

def NewMetricSpatialBeta(image: np.ndarray, beta: float=5.0, dt: float=.1, iterno: int=3,
                         adaptive: bool=True, mode: str="global_equalise") -> np.ndarray:
    current=np.asarray(image,float).copy()
    if not adaptive:
        beta_field=np.full_like(current,float(beta))
    elif mode=="global_equalise":
        beta_field=spatial_beta_field(current,beta)
    elif mode=="local_reliability":
        beta_field=local_reliability_beta_field(current,beta)
    else:
        raise ValueError(f"Unknown spatial-beta mode: {mode}")
    B=beta_field**2; Bx,By=_first(B); v1,v2=_direction(current)
    for _ in range(int(iterno)):
        Ix,Iy,Ixx,Iyy,Ixy=_derivatives(current); z=Ix**2+Iy**2; detg=1+B*z
        g11=1+B*Ix**2; g12=B*Ix*Iy; g22=1+B*Iy**2
        g11k1=Bx*Ix**2+2*B*Ixx*Ix
        g12k1=Bx*Ix*Iy+B*(Ixx*Iy+Ixy*Ix)
        g22k1=Bx*Iy**2+2*B*Ixy*Iy
        g11k2=By*Ix**2+2*B*Ixy*Ix
        g12k2=By*Ix*Iy+B*(Ixy*Iy+Iyy*Ix)
        g22k2=By*Iy**2+2*B*Iyy*Iy
        zk1=2*(Ix*Ixx+Iy*Ixy); zk2=2*(Ix*Ixy+Iy*Iyy)
        c=B*z/(detg+1e-12)
        ck1=(Bx*z+B*zk1)/(detg**2+1e-12); ck2=(By*z+B*zk2)/(detg**2+1e-12)
        vq=g11*v1**2+2*g12*v1*v2+g22*v2**2; K=1+c*vq; S=-c/(K+1e-12)
        invd=1/(detg+1e-12); gi00=invd*g22+S*v1*v1; gi01=-invd*g12+S*v1*v2; gi11=invd*g11+S*v2*v2
        vk1=g11*v1+g12*v2; vk2=g12*v1+g22*v2; vn=vk1**2+vk2**2
        a00=g11k1+ck1*vn; a01=g12k1+ck1*vn; a11=g22k1+ck1*vn
        b00=g11k2+ck2*vn; b01=g12k2+ck2*vn; b11=g22k2+ck2*vn
        k111=.5*gi00*a00+.5*gi01*(2*a01-b00); k112=.5*gi00*b00+.5*gi01*a11
        k122=.5*gi00*(2*b01-a11)+.5*gi01*b11; k211=.5*gi01*a00+.5*gi11*(2*a01-b00)
        k212=.5*gi01*b00+.5*gi11*a11; k222=.5*gi01*(2*b01-a11)+.5*gi11*b11
        delta=(gi00*(Ixx-k111*Ix-k211*Iy)+2*gi01*(Ixy-k112*Ix-k212*Iy)+gi11*(Iyy-k122*Ix-k222*Iy))
        current=current+float(dt)*delta
    return current

__all__=["NewMetricSpatialBeta","spatial_beta_field","local_reliability_beta_field"]
