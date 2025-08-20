import os.path as pa

import pandas as pd

from .paths import DATADIR

TABLEDIR = pa.join(DATADIR, "graham23_tables")

# Get GW catalog information
DF_GW = pd.read_csv(f"{TABLEDIR}/graham23_table1.plus.dat", sep="\s+")

# Get GW-flare association information
DF_ASSOC = pd.read_csv(f"{TABLEDIR}/graham23_table3.plus.dat", sep="\s+")

# Extract ra, dec from flare names
flareras = []
flaredecs = []
for f in DF_ASSOC["flarename"]:
    # RA
    rastr = f[1:10]
    ra = float(rastr[0:2]) + float(rastr[2:4]) / 60.0 + float(rastr[4:]) / 3600.0
    ra *= 360 / 24  # Convert to degrees
    flareras.append(ra)

    # Dec
    decstr = f[10:]
    dec = float(decstr[0:3]) + float(decstr[3:5]) / 60.0 + float(decstr[5:]) / 3600.0
    flaredecs.append(dec)
DF_ASSOC["flare_ra"] = flareras
DF_ASSOC["flare_dec"] = flaredecs

# Get flare information
# Columns not identical across GW events are dropped
DF_FLARE = DF_ASSOC.drop_duplicates(subset=["flarename"]).drop(
    columns=["gweventname", "ConfLimit", "vk_max"]
)

# Get bright? GW information
DF_GWBRIGHT = pd.read_csv(f"{TABLEDIR}/graham23_table4.plus.dat", sep="\s+")

# Get background flare information
DF_ASSOCPARAMS = pd.read_csv(f"{TABLEDIR}/graham23_table5.plus.dat", sep="\s+")

# Add gwtc data to graham23 table 1
DF_GWTC = pd.read_csv(f"{DATADIR}/gwtc/events.csv")
matchrows = []
for i, row in DF_GW.iterrows():
    # GW200105_162426 is not in the table
    if row["gweventname"] == "GW200105_162426":
        print("Copying custom data for GW200105_162426")
        dummyrow = dict(zip(DF_GWTC.columns, [None] * DF_GWTC.shape[1]))
        # Data to add (source: updated parameters in https://gwosc.org/eventapi/html/GWTC-3-marginal/GW200105_162426/v2/)
        adddata = {
            "name": "GW200105_162426",
            "mass_1_source": 9.1,
            "mass_1_source_lower": -1.7,
            "mass_1_source_upper": 1.7,
            "mass_2_source": 1.91,
            "mass_2_source_lower": -0.24,
            "mass_2_source_upper": 0.33,
            "luminosity_distance": 270,
            "luminosity_distance_lower": -110,
            "luminosity_distance_upper": 120,
            "chi_eff": 0.00,
            "chi_eff_lower": -0.18,
            "chi_eff_upper": 0.13,
            "total_mass_source": 11.0,
            "total_mass_source_lower": -1.4,
            "total_mass_source_upper": 1.5,
            "chirp_mass_source": 3.42,
            "chirp_mass_source_lower": -0.08,
            "chirp_mass_source_upper": 0.08,
            "chirp_mass": None,
            "chirp_mass_lower": None,
            "chirp_mass_upper": None,
            "final_mass_source": 10.8,
            "final_mass_source_lower": -1.4,
            "final_mass_source_upper": 1.5,
        }
        dummyrow.update(adddata)
        matchrows.append(dummyrow)
    elif row["gweventname"].strip("*") in DF_GWTC["name"].values:
        matchrow = DF_GWTC.loc[
            DF_GWTC["name"] == row["gweventname"].strip("*")
        ].to_dict(orient="records")[0]
        matchrows.append(matchrow)
    else:
        print(f"Did not find {row['gweventname']} in gwtc")
df_match = pd.DataFrame(matchrows)
DF_GWPLUS = pd.concat(
    [
        DF_GW,
        df_match,
    ],
    axis=1,
)
