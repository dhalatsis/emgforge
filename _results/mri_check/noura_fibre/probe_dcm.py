import pydicom, glob, collections, os
root="/home/noura/Documents/Projects/PhD/mri/data/SantoshS_MScProject"
descs=collections.Counter(); diff=[]
files=glob.glob(root+"/**/*.dcm",recursive=True)
print("total dcm:",len(files))
for f in files[:2000]:
    try: ds=pydicom.dcmread(f,stop_before_pixels=True,force=True)
    except Exception: continue
    sd=str(getattr(ds,"SeriesDescription","?"))
    seq=str(getattr(ds,"SequenceName",""))
    descs[(sd,seq)]+=1
    # diffusion indicators
    tags=[]
    for t,name in [(0x0018,"ScanningSequence"),(0x0043,"private"),(0x0019,"private")]:
        pass
    hit = any(k in (sd+seq).lower() for k in ["dwi","dti","diff","ep2d_diff","b400","b0","trace","adc","tensor"])
    bval=getattr(ds,"DiffusionBValue",None)
    if hit or bval is not None: diff.append((os.path.basename(f),sd,seq,bval))
print("\n=== series (description, sequence) : count ===")
for (sd,seq),n in descs.most_common(30): print(f"  {n:4d}  {sd!r} / {seq!r}")
print("\n=== diffusion-looking files ===")
for x in diff[:20]: print("  ",x)
print("  total diffusion-flagged:",len(diff))
