import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# DA-OB
k = [5000, 10000, 15000, 20000, 25000]
tau = [0.1, 0.15, 0.2, 0.25, 0.3]

default_k = 20000
default_tau = 0.2

varying_k_time_greedy = [3.052, 5.186, 7.707, 10.906, 16.249]
varying_k_time_stratified = [428.891, 432.201, 436.61, 442.785, 435.77]
varying_k_time_cost_based = [430.762, 441.105, 438.904, 446.247, 441.582]

varying_tau_time_greedy = [12.594, 11.813, 10.983, 10.538, 10.218]
varying_tau_time_stratified = [433.47, 442.785, 442.785, 442.785, 442.785]
varying_tau_time_cost_based = [444.816, 446.99, 446.247, 447.451, 441.108]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, OverlapBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DA_OB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, OverlapBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DA_OB_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# DA-DB
k = [2000, 3000, 4000, 5000, 6000]
tau = [0.1, 0.15, 0.2, 0.25, 0.3]

default_k = 3000
default_tau = 0.2

varying_k_time_greedy = [1.15, 1.293, 2.823, 5.137, 8.321]
varying_k_time_stratified = [419.55, 417.82, 419.35, 417.63, 427.26]
varying_k_time_cost_based = [429.272, 430.58, 429.413, 428.712, 429.915]

varying_tau_time_greedy = [19.176, 1.33, 1.318, 1.359, 1.25]
varying_tau_time_stratified = [419.35, 419.35, 419.35, 419.35, 419.35]
varying_tau_time_cost_based = [430.322, 429.634, 429.413, 431.12, 430.991]


fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, DeepBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DA_DB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, DeepBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DA_DB_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# DA-SW
k = [2000, 3000, 4000, 5000, 6000]
tau = [0.1, 0.15, 0.2, 0.25, 0.3]

default_k = 3000
default_tau = 0.1

varying_k_time_greedy = [1.527, 4.303, 17.031, 23.777, 57.834]
varying_k_time_stratified = [439.171, 436.28, 439.94, 436.307, 446.746]
varying_k_time_cost_based = [439.268, 437.833, 436.689, 437.811, 448.063]

varying_tau_time_greedy = [4.362, 1.66, 1.387, 1.387, 1.387]
varying_tau_time_stratified = [436.28, 436.28, 436.28, 436.28, 436.28]
varying_tau_time_cost_based = [437.833, 431.052, 437.129, 437.129, 437.129]


fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, Sudowoodo, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DA_SW_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-ACM, Sudowoodo, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DA_SW_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# Music-OB
k = [10000, 20000, 30000, 40000, 50000]
tau = [1.0, 2.0, 3.0, 4.0, 5.0]

default_k = 30000
default_tau = 3.0

varying_k_time_greedy = [6.337, 10.474, 15.337, 19.308, 23.53]
varying_k_time_stratified = [433.532, 438.491, 456.462, 484.296, 540.788]
varying_k_time_cost_based = [451.298, 478.362, 492.134, 460.71, 480.501]

varying_tau_time_greedy = [18.053, 16.153, 15.173, 15.002, 14.858]
varying_tau_time_stratified = [456.462, 456.462, 456.462, 456.462, 456.462]
varying_tau_time_cost_based = [856.62, 522.922, 492.134, 462.919, 452.889]


fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, OverlapBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/Music_OB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, OverlapBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/Music_OB_tau_vs_Time.png", dpi=300)
# ---------------------------------------------------------------
# Music_DB

k = [5000, 10000, 15000, 20000, 30000]
tau = [1.0, 2.0, 3.0, 4.0, 5.0]

default_k = 15000
default_tau = 3.0

varying_k_time_greedy = [479.413, 2050.442, 3312.865, 4020.367, 4888.561]
varying_k_time_stratified = [438.51, 437.003, 438.962, 447.172, 462.882]
varying_k_time_cost_based = [439.379, 466.148, 493.512, 498.65, 484.014]

varying_tau_time_greedy = [12738.57, 4403.815, 3312.865, 2747.283, 2265.999]
varying_tau_time_stratified = [438.962, 438.962, 438.962, 438.962, 438.962]
varying_tau_time_cost_based = [758.484, 570.926, 493.512, 451.627, 441.488]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, DeepBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/Music_DB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, DeepBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/Music_DB_tau_vs_Time.png", dpi=300)
# ---------------------------------------------------------------
# Music_SW
k = [2000, 4000, 6000, 8000, 10000]
tau = [0.1, 0.15, 0.2, 0.25, 0.3]

default_k = 6000
default_tau = 0.2

varying_k_time_greedy = [1.396, 232.213, 1091.536, 1110.921, 1222.351]
varying_k_time_stratified = [434.914, 433.784, 439.618, 438.32, 440.167]
varying_k_time_cost_based = [433.052, 446.044, 485.274, 545.971, 617.834]

varying_tau_time_greedy = [1533.49, 1268.659, 1091.536, 887.97, 685.754]
varying_tau_time_stratified = [439.618, 439.618, 439.618, 439.618, 439.618]
varying_tau_time_cost_based = [489.235, 483.902, 485.274, 483.236, 480.877]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, Sudowoodo, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/Music_SW_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("MusicBrainz, Sudowoodo, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/Music_SW_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# DS-OB
k = [100000, 125000, 150000, 175000, 200000]
tau = [0.05, 0.1, 0.15, 0.2, 0.25]

default_k = 150000
default_tau = 0.1

varying_k_time_greedy = [168.53, 295.975, 890.873, 867.772, 802.254]
varying_k_time_stratified = [1216.061, 1721.8, 2301.403, 3306.581, 3798.126]
# varying_k_time_cost_based = []

varying_tau_time_greedy = [862.026, 890.873, 449.903, 331.666, 276.397]
varying_tau_time_stratified = [2301.403, 2301.403, 2301.403, 2301.403, 2301.403]
# varying_tau_time_cost_based = []

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
# plt.plot(
#     k,
#     varying_k_time_cost_based,
#     label="Cost-based",
#     marker="o",
#     linestyle=":",
#     linewidth=2,
# )
plt.legend()
plt.title(
    "DBLP-Scholar, OverlapBlocker, Varying K vs. Time, \u03C4=" + str(default_tau)
)
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DS_OB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
# plt.plot(
#     tau,
#     varying_tau_time_cost_based,
#     label="Cost-based",
#     marker="o",
#     linestyle=":",
#     linewidth=2,
# )
plt.legend()
plt.title("DBLP-Scholar, OverlapBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DS_OB_tau_vs_Time.png", dpi=300)
# ---------------------------------------------------------------
# DS-DB
k = [5000, 10000, 15000, 20000, 25000]
tau = [0.2, 0.25, 0.3, 0.35, 0.4]

default_k = 20000
default_tau = 0.3

varying_k_time_greedy = [1.394, 229.5, 862.5, 4380, 9857.5]
varying_k_time_stratified = [492.632, 506.614, 535.361, 556.922, 576.856]
varying_k_time_cost_based = [589.99, 729.277, 981.392, 1350.384, 2068.31]

varying_tau_time_greedy = [8140.66, 6314.66, 4380, 2510, 1711.5]
varying_tau_time_stratified = [556.922, 556.922, 556.922, 556.922, 556.922]
varying_tau_time_cost_based = [1883, 1580.519, 1350.384, 1147.144, 1011.506]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-Scholar, DeepBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DS_DB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-Scholar, DeepBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DS_DB_tau_vs_Time.png", dpi=300)
# ---------------------------------------------------------------
# DS_SW

k = [4000, 8000, 12000, 16000, 20000]
tau = [0.05, 0.1, 0.15, 0.20, 0.25]

default_k = 12000
default_tau = 0.1

varying_k_time_greedy = [261.3, 634.283, 795.817, 912.308, 1611.463]
varying_k_time_stratified = [490.498, 523.786, 525.403, 537.17, 565.533]
varying_k_time_cost_based = [1057.847, 1095.041, 1463.837, 2438.512, 4293.574]

varying_tau_time_greedy = [509.955, 295.817, 212.372, 150.596, 41.311]
varying_tau_time_stratified = [523.786, 523.786, 523.786, 518.89, 523.786]
varying_tau_time_cost_based = [1745.255, 1463.837, 1302.412, 1176.112, 1124.457]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-Scholar, Sudowoodo, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/DS_SW_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("DBLP-Scholar, Sudowoodo, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/DS_SW_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# WA-OB
k = [200000, 250000, 300000, 350000, 400000]
tau = [2.5, 5.0, 7.5, 10.0, 12.5]

default_k = 300000
default_tau = 5.0

varying_k_time_greedy = [631.443, 958.32, 912.906, 985.414, 1021.352]
varying_k_time_stratified = [2923.964, 3978.144, 5706.51, 7700.185, 9818.87]
# varying_k_time_cost_based = []

varying_tau_time_greedy = [1017.891, 912.906, 837.564, 813.51, 804.985]
varying_tau_time_stratified = [9987.437, 5509.586, 5509.586, 5509.586, 5638.816]
# varying_tau_time_cost_based = []

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
# plt.plot(
#     k,
#     varying_k_time_cost_based,
#     label="Cost-based",
#     marker="o",
#     linestyle=":",
#     linewidth=2,
# )
plt.legend()
plt.title(
    "Walmart-Amazon, OverlapBlocker, Varying K vs. Time, \u03C4=" + str(default_tau)
)
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/WA_OB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
# plt.plot(
#     tau,
#     varying_tau_time_cost_based,
#     label="Cost-based",
#     marker="o",
#     linestyle=":",
#     linewidth=2,
# )
plt.legend()
plt.title(
    "Walmart-Amazon, OverlapBlocker, Varying \u03C4 vs. Time, K=" + str(default_k)
)
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/WA_OB_tau_vs_Time.png", dpi=300)

# ---------------------------------------------------------------
# WA-DB

k = [10000, 20000, 30000, 40000, 50000]
tau = [5.0, 7.5, 10.0, 12.5, 15.0]

default_k = 30000
default_tau = 10.0

varying_k_time_greedy = [341.28, 1046.913, 1735.949, 1659.208, 2461.963]
varying_k_time_stratified = [448.247, 462.168, 491.987, 537.82, 648.939]
varying_k_time_cost_based = [458.841, 477.63, 502.586, 520.055, 555.541]

varying_tau_time_greedy = [19430.212, 2707.46, 1735.949, 1157.434, 858.751]
varying_tau_time_stratified = [491.987, 491.987, 491.877, 491.139, 485.29]
varying_tau_time_cost_based = [1164.933, 573.187, 502.586, 490.523, 483.884]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("Walmart-Amazon, DeepBlocker, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/WA_DB_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("Walmart-Amazon, DeepBlocker, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/WA_DB_tau_vs_Time.png", dpi=300)
# ---------------------------------------------------------------
# WA-SW

k = [5000, 10000, 15000, 20000, 25000]
tau = [0.5, 1.0, 1.5, 2.0, 2.5]

default_k = 15000
default_tau = 1.0

varying_k_time_greedy = [103.894, 428.557, 788.745, 923.036, 842.392]
varying_k_time_stratified = [478.575, 746.64, 1311.465, 2132.083, 3138.045]
varying_k_time_cost_based = [630.862, 1108.544, 1925.446, 3037.673, 4255.501]

varying_tau_time_greedy = [1274.19, 788.745, 438.324, 231.863, 144.152]
varying_tau_time_stratified = [1778.775, 1311.465, 457.054, 688.077, 517.008]
varying_tau_time_cost_based = [2949.968, 1925.446, 1380.379, 1058.17, 828.95]

fig = plt.figure(figsize=(6, 6))
plt.plot(
    k, varying_k_time_greedy, label="Greedy", marker="o", linestyle="--", linewidth=2
)
plt.plot(
    k,
    varying_k_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    k,
    varying_k_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("Walmart-Amazon, Sudowoodo, Varying K vs. Time, \u03C4=" + str(default_tau))
plt.xlabel("K (Candidate Set Size)")
plt.ylabel("Time (Sec)")
plt.xticks(k)
plt.savefig("plots/WA_SW_K_vs_Time.png", dpi=300)

fig = plt.figure(figsize=(6, 6))
plt.plot(
    tau,
    varying_tau_time_greedy,
    label="Greedy",
    marker="o",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_stratified,
    label="Stratified",
    marker="o",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    tau,
    varying_tau_time_cost_based,
    label="Cost-based",
    marker="o",
    linestyle=":",
    linewidth=2,
)
plt.legend()
plt.title("Walmart-Amazon, Sudowoodo, Varying \u03C4 vs. Time, K=" + str(default_k))
plt.xlabel("\u03C4 (Fairness Threshold)")
plt.ylabel("Time (Sec)")
plt.xticks(tau)
plt.savefig("plots/WA_SW_tau_vs_Time.png", dpi=300)
