import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Constants
LUMINOSITY = 137000
c_signal = 1
c_background = 70

def calculate_weights(file_path):
    with uproot.open(file_path) as file:
        tree = file["ZZTree/candTree;1"]  # Correct tree location
        counters = file["ZZTree/Counters;1"]  # Correct Counters histogram location

        # Extract branches
        xsec = tree["xsec"].array()
        overall_event_weight = tree["overallEventWeight"].array()
        p_gg_sig = tree["p_GG_SIG_ghg2_1_ghz1_1_JHUGen"].array()
        p_qq_bkg = tree["p_QQB_BKG_MCFM"].array()

        # Denominator for weights (40th bin of Counters histogram)
        total_event_weight = counters.values()[39]

        # Compute weights
        weights = LUMINOSITY * xsec * overall_event_weight / total_event_weight

        return p_gg_sig, p_qq_bkg, weights

# Load signal and background samples
signal_file = "/home/public/data/ggH125/ZZ4lAnalysis.root"
background_file = "/home/public/data/qqZZ/ZZ4lAnalysis.root"

p_gg_sig_signal, p_qq_bkg_signal, weights_signal = calculate_weights(signal_file)
p_gg_sig_background, p_qq_bkg_background, weights_background = calculate_weights(background_file)

# Calculate D_bkg_kin for signal and background
D_bkg_kin_signal = 1 / (1 + c_signal * p_qq_bkg_signal / p_gg_sig_signal)
D_bkg_kin_background = 1 / (1 + c_background * p_qq_bkg_background / p_gg_sig_background)

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(D_bkg_kin_signal, bins=100, weights=weights_signal, alpha=0.5, color='blue', density=True, label='Signal')
plt.hist(D_bkg_kin_background, bins=100, weights=weights_background, alpha=0.5, color='red', density=True, label='Background')
plt.xlabel('$D^{bkg}_{kin}$')
plt.ylabel('Normalized Density')
plt.title('Weighted Histograms of $D^{bkg}_{kin}$')
plt.legend()
plt.savefig("Weighted_hist")

# ROC Curve (Threshold Method)
thresholds = np.linspace(0, 1, 1001)
TPR = []
FPR = []

for threshold in thresholds:
    tpr = np.sum((D_bkg_kin_signal > threshold) * weights_signal) / np.sum(weights_signal)
    fpr = np.sum((D_bkg_kin_background > threshold) * weights_background) / np.sum(weights_background)
    TPR.append(tpr)
    FPR.append(fpr)

plt.figure(figsize=(8, 6))
plt.plot(FPR, TPR, label='ROC Curve (Threshold Method)', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Threshold Method)')
plt.legend()
plt.savefig("threshold")

# ROC Curve (Histogram Integration)
hist_signal, bin_edges_signal = np.histogram(D_bkg_kin_signal, bins=100, weights=weights_signal, density=True)
hist_background, bin_edges_background = np.histogram(D_bkg_kin_background, bins=100, weights=weights_background, density=True)

TPR_hist = np.cumsum(hist_signal[::-1])[::-1] / np.sum(hist_signal)
FPR_hist = np.cumsum(hist_background[::-1])[::-1] / np.sum(hist_background)

plt.figure(figsize=(8, 6))
plt.plot(FPR_hist, TPR_hist, label='ROC Curve (Histogram Integration)', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Histogram Integration)')
plt.legend()
plt.savefig("histogram")

# Calculate AUC
auc_threshold = simpson(TPR, x=FPR)  # Explicitly pass x and y
auc_histogram = simpson(TPR_hist, x=FPR_hist)  # Explicitly pass x and y

print(f"AUC (Threshold Method): {auc_threshold:.4f}")
print(f"AUC (Histogram Integration): {auc_histogram:.4f}")
