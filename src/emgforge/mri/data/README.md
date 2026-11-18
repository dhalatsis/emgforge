# Forearm segmentation, subject S01

A voxel map of one forearm's tissues and muscles, segmented from MRI, for building subject-specific volume conductors
and simulating surface EMG. The MRI itself is not included.

| file | contents |
|---|---|
| `forearm_S01_segmentation.nii.gz` | label map, uint16: one label per voxel (bone, fat, 21 muscles, fiducial markers) |
| `forearm_S01_labels.json` | the label key: ID → name, tissue kind, abbreviation, compartment |

**Geometry.** 320 × 320 × 57 voxels of 0.4375 × 0.4375 × 6.0 mm (140 × 140 × 342 mm), with the affine in the NIfTI
header. The slices are axial and run along the forearm. The source scan was proton-density-weighted and
fat-saturated.

**Labels.** Labels follow ITK-SNAP integer conventions:
- 0: background;
- 1: fiducial markers (treat as background);
- 2, 3: radius and ulna (bone);
- 15: internal fat;
- 25: skin and subcutaneous fat;
- all other labels: muscles, named in `forearm_S01_labels.json`. FCU is 8, FDS 17, palmaris longus 14, and ECRL and
  ECRB share label 12.

**Status in the code.** emgforge does not load these files yet. `emgforge.mri.pipeline.DEFAULT_SEG` still names the
previous subject's file (removed), and the muscle atlas's label map is keyed to that subject's IDs. Until the code is
updated, pass this segmentation explicitly. Where you need atlas muscle data, map FDS from label 17 and palmaris longus
from 14.

**Consent.** Shared by the person scanned, who consents to its public release.

**Licence.** CC BY 4.0.
