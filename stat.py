import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from scipy.stats import mannwhitneyu

# Load CSV data
file_path = 'Stat.csv'
df = pd.read_csv(file_path)

# Extract values
x = df["x"].values
y = df["y"].values
labels = df["Label"].values

# Plot
plt.figure(figsize=(8, 6))

for label in pd.unique(labels):
    idx = labels == label
    if label==1:
        label_txt="Pattern B"
    if label==-1:
        label_txt="Pattern A"
    plt.scatter(x[idx], y[idx], label=f" {label_txt}", s=60)
    plt.xscale('symlog', linthresh=1e-5)  
    plt.yscale('symlog', linthresh=.0001) 
# Thick black lines through origin
plt.axhline(y=0, color='black', linewidth=2)
plt.axvline(x=0, color='black', linewidth=2)

#plt.xlabel("PSD, RL-MI, left hemi - PSD, LL-MI, left hemi",fontsize=18)
plt.xlabel(r"$\mathrm{PSD}_{\mathrm{RL\text{-}MI}} - \mathrm{PSD}_{\mathrm{LL\text{-}MI}} \text{ at LH}$", fontsize=18)
plt.ylabel(r"$\mathrm{PSD}_{\mathrm{RL\text{-}MI}} - \mathrm{PSD}_{\mathrm{LL\text{-}MI}} \text{ at RH}$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(fr"final_both_sets")
plt.show()


aa=df[df['Label'] == -1]['y'].values
stat_A, p_A =  ttest_1samp(df[df['Label'] == -1]['y'].values, popmean=0, alternative='greater')

bb=df[df['Label'] == 1]['x'].values
stat_B, p_B =  ttest_1samp(df[df['Label'] == 1]['x'].values, popmean=0, alternative='less')


stop=1


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.patches import Patch

# # Data
# target_2a = [88.89,81.94,87.5,84.72,83.33,84.72,88.88,86.11,87.5,76.38]
# random_2a = [87.5,81.94,84.72,76.38,83.33,83.33,88.88,69.44,79.16,81.94]
# target_pn = [81,73,87,73,73,70,75,78,80,64]
# random_pn = [81,73,58,85,75,85,58,77,63,62]

# variance_1 = np.var(target_2a, ddof=1)
# variance_2 = np.var(random_2a, ddof=1)
# variance_3 = np.var(target_pn, ddof=1)
# variance_4 = np.var(random_pn, ddof=1)

# me_1 = np.mean(target_2a)
# me_2 = np.mean(random_2a)
# me_3 = np.mean(target_pn)
# me_4 = np.mean(random_pn)

# ma_1 = np.max(target_2a)
# ma_2 = np.max(random_2a)
# ma_3 = np.max(target_pn)
# ma_4 = np.max(random_pn)

# # Assign numeric positions for boxplots
# data = [
#     (0.9, target_2a, 'Cluster'),
#     (1.1, random_2a, 'Random'),
#     (1.5, target_pn, 'Cluster'),
#     (1.7, random_pn, 'Random')
# ]

# # Color mapping
# colors = {'Cluster': '#4c72b0', 'Random': '#55a868'}

# # Plot
# plt.figure(figsize=(10, 6))
# for x, values, label in data:
#     plt.boxplot(values, positions=[x], widths=0.15, patch_artist=True,
#                 boxprops=dict(facecolor=colors[label]))

# # Formatting
# plt.xticks([1, 1.6], ['BCI 2a', 'PhysioNet'],fontsize=18)
# plt.ylabel("Accuracy [%]",fontsize=18)
# plt.xlabel("",fontsize=18)

# # Legend
# legend_handles = [Patch(color=colors['Cluster'], label='Cluster'),
#                   Patch(color=colors['Random'], label='Random')]
# plt.legend(handles=legend_handles, fontsize=18)

# plt.tight_layout()
# #plt.savefig("tight_spacing_with_legend.png")
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (replace with your actual filename)
df = pd.read_csv(fr'ERS.csv')  # Replace with your CSV file path

# Define column pairs for plotting
pairs = [
    ('Pattern B_PN', 'Other_1_PN'),
    ('Pattern A_PN', 'Other_2_PN'),
    ('Pattern B_2a', 'Other_1_2a'),
    ('Pattern A_2a', 'Other_2_2a')
]



for col1, col2 in pairs:
    plt.figure(figsize=(4, 6))
    data = [df[col1].dropna(), df[col2].dropna()]

    stat, p = mannwhitneyu(data[0], data[1], alternative='two-sided')
    print(f"{col1} vs {col2} — Mann–Whitney U test p-value: {p:.4f}")

    # Create boxplot
    box = plt.boxplot(
        data,
        positions=[1, 1.4],
        widths=0.3,
        patch_artist=True,
        labels=['', '']
    )

    # Set colors
    colors = ['lightyellow', 'lightblue']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Annotate medians in black
    for i, dataset in enumerate(data):
        median = dataset.median()
        plt.text(1 + i * 0.4, median, f'MD: {median:.2f}', ha='center', va='bottom', fontsize=11, color='black')

    plt.xticks([], fontsize=12)
    plt.xlabel('')
    plt.tight_layout()
    plt.show()




stop=1