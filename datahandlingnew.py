import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
FILE_PATH = "events_anomalydetection_v2.features.h5"
 
df = pd.read_hdf(FILE_PATH)
 
label_col = "label"

# COMPUTING DERIVED QUANTITIES
 
for j in ["j1", "j2"]:
    px, py, pz = df[f"px{j}"], df[f"py{j}"], df[f"pz{j}"]
    m = df[f"m{j}"]
 
    # transverse momentum
    df[f"pt{j}"] = np.sqrt(px**2 + py**2)
 
    # energy from on-shell relation: E^2 = p^2 + m^2
    p2 = px**2 + py**2 + pz**2
    df[f"E{j}"] = np.sqrt(p2 + m**2)
 
    # n-subjettiness ratio (2-prong-ness): tau2/tau1
    df[f"tau21{j}"] = df[f"tau2{j}"] / df[f"tau1{j}"]
 
    # n-subjettiness ratio (3-prong-ness): tau3/tau2
    df[f"tau32{j}"] = df[f"tau3{j}"] / df[f"tau2{j}"]
 
# dijet invariant mass: combine the two jets' 4-vectors
E_tot = df["Ej1"] + df["Ej2"]
px_tot = df["pxj1"] + df["pxj2"]
py_tot = df["pyj1"] + df["pyj2"]
pz_tot = df["pzj1"] + df["pzj2"]
 
df["mjj"] = np.sqrt(
    np.clip(E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2), 0, None)
)

for j in ["j1", "j2"]:
    px, py, pz = df[f"px{j}"], df[f"py{j}"], df[f"pz{j}"]
    p = np.sqrt(px**2 + py**2 + pz**2)
 
    # pseudorapidity: eta = arctanh(pz / p)
    df[f"eta{j}"] = np.arctanh(np.clip(pz / p, -0.999999, 0.999999))
 
    # azimuthal angle
    df[f"phi{j}"] = np.arctan2(py, px)
 
# angular separation between the two jets
d_eta = df["etaj1"] - df["etaj2"]
d_phi = df["phij1"] - df["phij2"]
# wrap delta-phi into [-pi, pi]
d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
 
df["delta_eta"] = np.abs(d_eta)
df["delta_phi"] = np.abs(d_phi)
df["delta_R"] = np.sqrt(d_eta**2 + d_phi**2)
 
# mass and pT asymmetry between jets
df["mass_asym"] = np.abs(df["mj1"] - df["mj2"]) / (df["mj1"] + df["mj2"])
df["pt_asym"] = np.abs(df["ptj1"] - df["ptj2"]) / (df["ptj1"] + df["ptj2"])
 
# system-level: total transverse momentum of the dijet system (vector sum)
sys_px = df["pxj1"] + df["pxj2"]
sys_py = df["pyj1"] + df["pyj2"]
df["system_pt"] = np.sqrt(sys_px**2 + sys_py**2)
 
# HT: scalar sum of jet pTs
df["HT"] = df["ptj1"] + df["ptj2"]
 
# cos(theta): production/decay angle in the dijet CM frame,
# approximated here via the standard dijet formula using rapidities
df["cos_theta_star"] = np.tanh(d_eta / 2)
 
# extra n-subjettiness ratio: 3-prong relative to 1-prong
for j in ["j1", "j2"]:
    df[f"tau31{j}"] = df[f"tau3{j}"] / df[f"tau1{j}"]
 
 
df["pt_lead"] = df[["ptj1", "ptj2"]].max(axis=1)
df = df[df["pt_lead"] > 1200].reset_index(drop=True)  # GeV, since masses/momenta are in GeV
 
df = df.dropna().reset_index(drop=True)

# DEFINING SIDEBAND VS SIGNAL REGION

mjj_col = "mjj"
 
signal_region = (df[mjj_col] > 3300) & (df[mjj_col] < 3700)
sideband_region = ~signal_region
 
df_sideband = df[sideband_region].reset_index(drop=True)
df_signal_region = df[signal_region].reset_index(drop=True)
 
print("\nSideband events:", len(df_sideband))
print("Signal-region events:", len(df_signal_region))
 
# TRAIN / VAL / TEST SPLIT
all_features= ["mj1", "mj2", "tau21j1", "tau21j2", "tau32j1", "tau32j2", "tau31j1", "tau31j2", 
               "tau1j1", "tau2j1", "tau3j1", "tau1j2", "tau2j2", "tau3j2", "ptj1", "ptj2", 
               "delta_eta", "delta_phi", "delta_R", "mass_asym", "pt_asym", "system_pt", "HT", "cos_theta_star",]

feature_cols = ["mj1", "mj2", "tau21j1", "tau21j2","tau32j1", "tau32j2", "tau31j1", "tau31j2",
                "tau1j1", "tau2j1", "tau3j1", "tau1j2", "tau2j2", "tau3j2", "ptj1", "ptj2",
                "delta_eta", "delta_phi", "delta_R", "cos_theta_star", "pt_asym", "system_pt",
                "mass_asym", "HT"] 

train_val_df = df_sideband
train_df, val_df = train_test_split(
    train_val_df, test_size=0.2, random_state=42
)
  

test_df = df_signal_region
 
print("\nTrain size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))
print("\nFEATURES: ", len(feature_cols), feature_cols)
 
# NORMALIZE FEATURES (fit scaler on train set only)
scaler = StandardScaler()
 
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)
 
Y_test = test_df[label_col].values 