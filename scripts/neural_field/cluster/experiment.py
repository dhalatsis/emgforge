"""The experiment definition, as data — one source of truth for the job matrix.

Both the job manifest AND the analysis import from here, so the training grid and the held-out
benchmark anatomies cannot drift apart. The benchmark anatomies (21_build_bench_anatomy's
HELDOUT_ANATOMY) sit at half-steps INTERIOR to this training grid -> interpolation test.

Pure data + a job enumerator: no torch, no dolfinx, importable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------------ the grids
# Cylinder anatomy conditioning (the varying-anatomy / 594x regime).
# r_fat is the OUTER fat boundary; the muscle boundary is fixed at r_muscle=35mm, so every
# r_fat MUST exceed 35 (fat thickness = r_fat - 35, here 1..11mm). r_fat<=35 is a degenerate
# fat shell that gmsh cannot build — 20_gen_cyl_anatomy guards it.
CYL_FAT = (36.0, 38.0, 40.0, 42.0, 44.0, 46.0)          # r_fat (mm); each is a NEW mesh
CYL_PEN = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)            # pennation (deg); rotates muscle sigma
CYL_N_ELEC = 96                                         # electrodes per anatomy
# held-out benchmark anatomies live at half-steps INTERIOR to these (see 21_build_bench_anatomy).
# Assert here so a grid edit that collides with the benchmark fails loudly.
HELDOUT_ANATOMY = ((37.0, 12.5), (39.0, 7.5), (41.0, 2.5), (43.0, 17.5))

# MRI cross-electrode (single anatomy, cheap WR solves) — the other headline axis.
MRI_N_ELEC = (256, 512, 1024, 2048)

# anatomy normalisation ranges the trainer/scorer must share (span the grid, keep FIXED).
FAT_LO, FAT_HI = 36.0, 46.0
PEN_LO, PEN_HI = 0.0, 25.0


@dataclass
class Job:
    name: str
    stage: str                       # gen_cyl | gen_mri | bench_cyl | bench_mri | train | score
    where: str                       # "cpu" (dolfinx FEM) | "gpu" (torch)
    cmd: list[str]
    needs: list[str] = field(default_factory=list)


def _py(script, *args):
    return ["python", f"scripts/neural_field/{script}", *map(str, args)]


def jobs() -> list[Job]:
    """The full DAG. CPU stages generate FEM data onto the shared FS; GPU stages train/score.

    Dependency shape:
        gen_cyl[fat,pen]  --.                          bench_cyl --.
        (36 FEM jobs)       >-- train_cyl -- score_cyl <-----------'
        bench_cyl (1 FEM) --'
        gen_mri[N] -- train_mri[N] -- score_mri[N]     (+ bench_mri, once)
    """
    J: list[Job] = []

    # ---- cylinder anatomy grid: one CPU FEM job per (fat, pennation) ----
    for f in CYL_FAT:
        for p in CYL_PEN:
            assert (f, p) not in HELDOUT_ANATOMY, f"train grid collides with benchmark at {(f,p)}"
            J.append(Job(f"gen_cyl_f{f:g}_p{p:g}", "gen_cyl", "cpu",
                         _py("20_gen_cyl_anatomy.py", "--r_fat", f, "--pennation", p,
                             "--n_elec", CYL_N_ELEC, "--out", "_results/neural_field/anat")))
    J.append(Job("bench_cyl", "bench_cyl", "cpu", _py("21_build_bench_anatomy.py")))

    train_cyl = Job("train_cyl", "train", "gpu",
                    _py("22_train_conditioned.py", "--data_glob",
                        "_results/neural_field/anat/cyl_fat*_pen*_n*.npz",
                        "--fat_lo", FAT_LO, "--fat_hi", FAT_HI,
                        "--pen_lo", PEN_LO, "--pen_hi", PEN_HI, "--tag", "grid"),
                    needs=[j.name for j in J if j.stage == "gen_cyl"])
    score_cyl = Job("score_cyl", "score", "gpu",
                    _py("23_score_conditioned.py", "--ckpt",
                        "_results/neural_field/cyl_cond_mlp_raw_grid.pt"),
                    needs=["train_cyl", "bench_cyl"])
    J += [train_cyl, score_cyl]

    # ---- MRI cross-electrode: gen (CPU) -> train (GPU) -> score (GPU) per N ----
    J.append(Job("bench_mri", "bench_mri", "cpu", _py("11_build_muap_bench_mri.py")))
    for n in MRI_N_ELEC:
        gen = Job(f"gen_mri_{n}", "gen_mri", "cpu", _py("12_gen_mri_dataset.py", "--n", n))
        tr = Job(f"train_mri_{n}", "train", "gpu",
                 _py("03_train_cyl.py", "--data",
                     f"_results/neural_field/mri_elec_{n}_near_muscle.npz",
                     "--target", "raw", "--tag", f"mriN{n}"), needs=[gen.name])
        sc = Job(f"score_mri_{n}", "score", "gpu",
                 _py("13_score_bench_mri.py", "--ckpt",
                     f"_results/neural_field/cyl_mlp_raw_mriN{n}.pt"),
                 needs=[tr.name, "bench_mri"])
        J += [gen, tr, sc]
    return J


if __name__ == "__main__":
    J = jobs()
    cpu = [j for j in J if j.where == "cpu"]; gpu = [j for j in J if j.where == "gpu"]
    print(f"{len(J)} jobs: {len(cpu)} CPU (FEM) + {len(gpu)} GPU (torch)")
    print(f"cylinder grid: {len(CYL_FAT)}×{len(CYL_PEN)} = {len(CYL_FAT)*len(CYL_PEN)} anatomies "
          f"× {CYL_N_ELEC} electrodes")
    print(f"MRI electrode sweep: N ∈ {MRI_N_ELEC}")
    for j in J:
        dep = f"  <- {', '.join(j.needs)}" if j.needs else ""
        print(f"  [{j.where}] {j.name} ({j.stage}){dep}")
