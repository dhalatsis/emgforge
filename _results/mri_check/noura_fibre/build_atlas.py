import json, pathlib
SRC="/home/noura/Documents/Projects/PhD/simulation/neural-forward-emg/mri/data/forearm_muscle_atlas.json"
atlas=json.load(open(SRC)); M=atlas["muscles"]

# --- NEW fields per muscle, from the literature scout ---
# Lf_Lm: Lieber&Friden 2000 / Liu 2014.  pennation_type: architecture thread (Lieber/Campisi/StatPearls).
# IZ_fraction (proximal=0..distal=1) + n_iz_bands: Safwat 2007, Saito 2000, Lateva 2010, Barbero-Merletti atlas.
NEW = {
 "brachioradialis":                dict(Lf_Lm=0.69, ptype="fusiform",     iz=[0.2,0.4,0.6,0.8], bands=4),  # series-fibred, 3-6 endplate zones (Lateva 2010)
 "flexor_carpi_radialis":          dict(Lf_Lm=0.27, ptype="bipennate",    iz=[0.17],            bands=1),
 "flexor_carpi_ulnaris":           dict(Lf_Lm=0.20, ptype="bipennate",    iz=[0.17,0.62],       bands=2),  # split-bipennate, +distal branch (Safwat)
 "palmaris_longus":                dict(Lf_Lm=0.36, ptype="fusiform",     iz=[0.17],            bands=1),
 "pronator_teres":                 dict(Lf_Lm=0.28, ptype="bipennate",    iz=[0.20,0.45],       bands=2),
 "flexor_pollicis_longus":         dict(Lf_Lm=0.24, ptype="unipennate",   iz=[0.17],            bands=1),
 "extensor_carpi_radialis_longus": dict(Lf_Lm=0.41, ptype="unipennate",   iz=[0.35],            bands=1),
 "extensor_carpi_radialis_brevis": dict(Lf_Lm=0.30, ptype="unipennate",   iz=[0.16],            bands=1),
 "extensor_carpi_ulnaris":         dict(Lf_Lm=0.28, ptype="bipennate",    iz=[0.17],            bands=1),
 "extensor_pollicis_longus":       dict(Lf_Lm=0.31, ptype="unipennate",   iz=[0.17],            bands=1),
 "extensor_pollicis_brevis":       dict(Lf_Lm=0.31, ptype="unipennate",   iz=[0.17],            bands=1),
 "extensor_indicis":               dict(Lf_Lm=0.32, ptype="unipennate",   iz=[0.50],            bands=1),  # mid-belly IZ (unusual; Safwat)
 "abductor_pollicis_longus":       dict(Lf_Lm=0.31, ptype="multipennate", iz=[0.17],            bands=2),
 "supinator":                      dict(Lf_Lm=0.40, ptype="unipennate",   iz=[0.13],            bands=1),  # wrapped flat sheet
}
# per-digit families share the base architecture
FAM = {
 "flexor_digitorum_superficialis": dict(Lf_Lm=0.30, ptype="multipennate", iz=[0.42,0.72],       bands=2),  # Campisi 2023 (3 aponeuroses)
 "flexor_digitorum_profundus":     dict(Lf_Lm=0.27, ptype="multipennate", iz=[0.15,0.30],       bands=2),
 "extensor_digitorum_communis":    dict(Lf_Lm=0.29, ptype="bipennate",    iz=[0.30],            bands=1),
}
UNCERTAIN={"flexor_carpi_radialis","extensor_carpi_radialis_brevis","flexor_digitorum_profundus",
           "flexor_pollicis_longus","supinator"}  # class source-dependent (see brief)

def apply(key,d):
    m=M[key]
    m["Lf_Lm_ratio"]=d["Lf_Lm"]
    m["pennation_type"]=d["ptype"]
    m["IZ_fraction"]=d["iz"]
    m["n_iz_bands"]=d["bands"]
    m["class_uncertain"]=key in UNCERTAIN
    # convenience: fibre length in mm (method uses mm)
    if "fascicle_length_cm" in m: m["fascicle_length_mm"]=round(m["fascicle_length_cm"]*10,1)

for k,d in NEW.items(): apply(k,d)
for k in M:
    for fam,d in FAM.items():
        if k.startswith(fam): apply(k,d)

# --- add muscles the atlas was missing (values: Holzbaur 2005 Lopt + Lieber where available) ---
def add(key,Lf_cm,penn,pcsa,d):
    M[key]={"fascicle_length_cm":Lf_cm,"fascicle_length_mm":round(Lf_cm*10,1),
            "pennation_deg":penn,"PCSA_cm2":pcsa,
            "Lf_Lm_ratio":d["Lf_Lm"],"pennation_type":d["ptype"],
            "IZ_fraction":d["iz"],"n_iz_bands":d["bands"],"class_uncertain":False,
            "source_note":"added from Holzbaur 2005 / architecture scout (not in original Lieber atlas)"}
if "pronator_quadratus" not in M:
    add("pronator_quadratus",2.3,10.0,6.3,dict(Lf_Lm=0.42,ptype="unipennate",iz=[0.87],bands=1))
if "extensor_digiti_minimi" not in M:
    add("extensor_digiti_minimi",5.9,3.0,1.6,dict(Lf_Lm=0.32,ptype="unipennate",iz=[0.17],bands=1))
if "anconeus" not in M:
    add("anconeus",3.0,0.0,2.5,dict(Lf_Lm=0.45,ptype="fusiform",iz=[0.5],bands=1))

# --- WR lab dataset label -> atlas key (from mri/data/Lab/WR/Labels.txt, raw ITK-SNAP) ---
atlas["wr_label_to_muscle"]={
 "4":"supinator","5":"anconeus","6":"extensor_carpi_ulnaris","7":"flexor_digitorum_profundus_middle",
 "8":"flexor_carpi_ulnaris","9":"extensor_digitorum_communis_middle","10":"extensor_digiti_minimi",
 "11":"brachioradialis","12":"extensor_carpi_radialis_longus","13":"flexor_digitorum_superficialis_middle",
 "14":"palmaris_longus","16":"flexor_carpi_radialis","18":"abductor_pollicis_longus","19":"extensor_indicis",
 "20":"extensor_pollicis_longus","21":"pronator_teres","22":"flexor_pollicis_longus","23":"pronator_quadratus"}

atlas["_meta"]["derived_from"]=(
 "Extended copy of neural-forward-emg/mri/data/forearm_muscle_atlas.json (original left untouched). "
 "Original fascicle_length_cm/pennation_deg/PCSA_cm2/mass_g are the Lieber 1990/1992 values kept as-is.")
atlas["_meta"]["extended_2026_07"]=(
 "Added, per muscle: Lf_Lm_ratio, pennation_type, IZ_fraction (proximal=0..distal=1), n_iz_bands, "
 "fascicle_length_mm, class_uncertain. Added muscles PQ/EDM/anconeus. Added wr_label_to_muscle. "
 "See field_sources for provenance. Caveat: cadaver motor-points sit slightly proximal to the true "
 "endplate band, and several IZ fractions are thirds-level (~0.17) pending the exact Barbero atlas table.")
atlas["_meta"]["field_sources"]={
 "fascicle_length_cm / fascicle_length_mm":"ORIGINAL atlas — Lieber et al. 1990 (J Hand Surg 15A:244) & 1992 (17A:787), cadaver + laser-diffraction",
 "pennation_deg":"ORIGINAL atlas — Lieber et al. 1990/1992",
 "PCSA_cm2 / mass_g":"ORIGINAL atlas — Lieber et al. 1990/1992",
 "Lf_Lm_ratio":"NEW — Lieber & Friden 2000 (Muscle & Nerve 23:1647, range 0.2-0.6); Liu et al. 2014 (PMC4089342, forearm compartments)",
 "pennation_type":"NEW — Lieber et al. 1992; Campisi et al. 2023 (J Anat, FDS 5-belly/3-aponeurosis); StatPearls (FCU/PT/ECU split-bipennate); Ravichandiran & Agur 2012 (ECRL/ECRB)",
 "IZ_fraction":"NEW — Safwat & Abdel-Meguid 2007 (Folia Morphol 66:83, forearm motor-points); Saito et al. 2000 (J Human Ergol, HD-sEMG); Lateva et al. 2010 (J Appl Physiol, brachioradialis 3-6 bands); Barbero, Merletti & Rainoldi 2012 (Atlas of Muscle Innervation Zones — exact forearm fractions paywalled, so several values are thirds-level)",
 "n_iz_bands":"NEW — same sources as IZ_fraction (BR 3-6 per Lateva; FDS 2 per Campisi; pennate singles = 1)",
 "added muscles (pronator_quadratus, extensor_digiti_minimi, anconeus)":"NEW — Holzbaur, Murray & Delp 2005 (Ann Biomed Eng 33:829, MoBL-ARMS) for Lf/pennation/PCSA; architecture scout for type/IZ",
 "wr_label_to_muscle":"NEW — mri/data/Lab/WR/Labels.txt (raw ITK-SNAP label indices for the WR lab subject)"}
atlas["_meta"]["units"]["IZ_fraction"]="fraction of muscle length, proximal end = 0"
atlas["_meta"]["units"]["Lf_Lm_ratio"]="fascicle length / muscle-belly length"

# --- write to emgforge data dir (where apply_atlas_pennation.py expects it) ---
dst=pathlib.Path("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data")
dst.mkdir(parents=True,exist_ok=True)
out=dst/"forearm_muscle_atlas.json"
json.dump(atlas,open(out,"w"),indent=2)
print("wrote",out,"—",len(M),"muscles")
# sanity print
for k in ("flexor_carpi_ulnaris","flexor_digitorum_superficialis_middle","brachioradialis","pronator_quadratus"):
    m=M[k]; print(f"  {k}: Lf={m['fascicle_length_mm']}mm Lf/Lm={m['Lf_Lm_ratio']} penn={m['pennation_deg']}deg "
                   f"type={m['pennation_type']} IZ={m['IZ_fraction']} bands={m['n_iz_bands']}")
