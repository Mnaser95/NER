import mne
from scipy.signal import stft
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.signal import welch
warnings.filterwarnings("ignore")

N_SUBJECT = 109
BASELINE_EYE_CLOSED = [2]
IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
SELECTED_CHANNELS = [8,9,1,2,15,16, 11,12,4,5,18,19]
num_chan_per_hemi=len(SELECTED_CHANNELS)//2
fs=160

num_runs_sub=3
needed_subs=109
low_rest=1
high_rest=20
low_MI=8
high_MI=12
needed_ratio=1.2

rest_plotting=False
MI_plotting=True

#################################################################################################################################################### Rest
def raw_rest_processing(raw):
    raw.filter(l_freq=low_rest, h_freq=high_rest, picks="eeg", verbose='WARNING')
    events, _ = mne.events_from_annotations(raw)

    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions

    epoched = mne.Epochs(raw,events,event_id=dict(rest=1),tmin=1,tmax=59,proj=False,picks=SELECTED_CHANNELS,baseline=None,preload=True)

    return epoched
def rest_data_generation(epoched):
    X = (epoched.get_data() * 1e3).astype(np.float32)

    avg_left = X[:, :num_chan_per_hemi, :].mean(axis=1)  # (n_samples, time)
    avg_right = X[:, -num_chan_per_hemi:, :].mean(axis=1)

    # Apply STFT per sample and average across samples
    Zxx1_total = []
    Zxx2_total = []
    f, _, Z1 = stft(avg_left[0], fs, nperseg=fs)
    f, _, Z2 = stft(avg_right[0], fs, nperseg=fs)
    Zxx1_total.append(np.abs(Z1))
    Zxx2_total.append(np.abs(Z2))

    Zxx1_mean = np.mean(Zxx1_total, axis=0)
    Zxx2_mean = np.mean(Zxx2_total, axis=0)
    data = (Zxx1_mean - Zxx2_mean).squeeze()

    return data
def subject_select(mid,other):
    segment_size = 10
    n_segments = len(mid) // segment_size
    #ratio = [mid[i]/other[i] for i in range(n_segments)]
    ratios = []
    votes=[]
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        avg_mid_abs = np.abs(np.mean(mid[start:end]))
        avg_other_abs = np.abs(np.mean(other[start:end]))
        ratio = avg_mid_abs / avg_other_abs
        if ratio > needed_ratio:
            if np.mean(mid[start:end])>0:
                votes.append("11") 
            else:
                votes.append("10")
        else:
            votes.append("0X")

    #"11": strong, positive (mainly yellow)
    #"10": strong, negative (mainly blue)
    #"0X": weak 


    vote_counts = Counter(votes)
    majority_vote, num_majority_votes = vote_counts.most_common(1)[0]


    if majority_vote=="11":
        res="Pattern B"
    if majority_vote=="10":
        res="Pattern A"
    if majority_vote=="0X":
        res="Weak"
    confidence=num_majority_votes/n_segments
    return res, confidence
def rest_plotting(data,res_down,confidence_down):
    # Define a custom colormap
    colors = [
        (0, 'blue'),
        (0.3, 'skyblue'),
        (0.5, 'black'),
        (0.7, 'lightyellow'),
        (1, 'yellow')
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        aspect='auto',
        cmap=custom_cmap,
        interpolation='nearest',
        vmin=-5e-6,
        vmax=5e-6
    )

    # Custom colorbar
    cbar = plt.colorbar(im)
    cbar.set_ticks([-5e-6, 0, 5e-6])
    cbar.set_ticklabels(["Min (-ve)", "0", "Max (+ve)"])
    cbar.ax.tick_params(labelsize=12)

    # Custom x-axis ticks (e.g., 5 ticks between 0 and 60)
    tick_labels = [0, 15, 30, 45, 60]
    tick_positions = [int(i * (data.shape[1] / 60)) for i in tick_labels]
    plt.xticks(tick_positions, [str(i) for i in tick_labels], fontsize=12)
    plt.yticks(fontsize=12)

    # Axis labels
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Frequency [Hz]", fontsize=14)

    # Optional: Add a title
    plt.title(fr"|STFT| left hemi - |STFT| right hemi", fontsize=16)

    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(fr"Rest_{sub}.png")
    plt.close()

# Download/load data paths
physionet_paths = [mne.datasets.eegbci.load_data(subject_id,BASELINE_EYE_CLOSED,"/root/mne_data" ) for subject_id in range(1, needed_subs + 1) ]
physionet_paths = np.concatenate(physionet_paths)

# Read EDF files
parts = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING')for path in physionet_paths] # 0-indexed

patterns={}
confidences={} # absolute
confidences_multi={} # both +ve and -ve

for sub, raw in enumerate(parts):
    epoched=raw_rest_processing(raw)

    data=rest_data_generation(epoched) # multiplied by 1000

    avg_mid_freq=data[7:13,:].mean(axis=0)
    avg_below_freq=data[2:7,:].mean(axis=0)
    avg_above_freq=data[13:18,:].mean(axis=0)

    res_down, confidence_down=subject_select(avg_mid_freq,avg_below_freq)

    patterns[sub+1] = res_down
    confidences[sub+1] = confidence_down

    if rest_plotting:
        rest_plotting(data,res_down,confidence_down)

for k in [k for k, v in patterns.items() if v == "Weak"]: #remove weak
    del patterns[k]
    del confidences[k]

for k in [38,88,89,92,100,104]: #remove incorrecrt datapoints
    if k in patterns:
        del patterns[k]
        del confidences[k]


subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
confidences_multi = {k: (v if patterns[k] == "Pattern B" else -v) for k, v in confidences.items()}

stop=1
#################################################################################################################################################### MI
def raw_MI_processing(raw):
    raw.filter(l_freq=low_MI, h_freq=high_MI, picks="eeg", verbose='WARNING')
    
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    
    df = pd.read_csv(fr"64montage.csv", header=None, names=["name", "x", "y", "z"])    
    ch_pos = {row['name']: [row['x'], row['y'], row['z']] for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage,on_missing="warn")
    valid_chs = [
    ch['ch_name'] for ch in raw.info['chs']
    if ch['loc'] is not None and not np.allclose(ch['loc'][:3], 0) and not np.isnan(ch['loc'][:3]).any()
    ]
    raw= raw.copy().pick_channels(valid_chs)

    events, _ = mne.events_from_annotations(raw)

    epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=0, tmax=4,proj=False, picks=SELECTED_CHANNELS, baseline=None, preload=True)
    
    a=epoched.get_data().shape[2]
    if a<630:
        stop=1
    return epoched,epoched.get_data()
def MI_data_generation(epoched):
    X = (epoched.get_data() * 1e3).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)

    _, _, Zxx = stft(X, fs, nperseg=fs)
    MI_tf = np.abs(Zxx)
    X_tf = MI_tf.mean(axis=2).mean(axis=2) # this will avg over time and freq

    f, psd_run = welch(X, fs=fs)
    psd_run_avg_freq=psd_run.mean(axis=2)
    psd_run_avg_freq_left = psd_run_avg_freq[:, :num_chan_per_hemi].mean(axis=1)
    psd_run_avg_freq_right = psd_run_avg_freq[:, -num_chan_per_hemi:].mean(axis=1)

    avg_left = X_tf[:, :num_chan_per_hemi].mean(axis=1)
    avg_right = X_tf[:, -num_chan_per_hemi:].mean(axis=1)
    return avg_left,avg_right,y,psd_run_avg_freq_left,psd_run_avg_freq_right
def res_plotting(x_vals,y_vals,keys,subs_pattern_B):
    # Plot
    plt.clf()

    # Add x=0 and y=0 lines
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    # plt.xscale('symlog', linthresh=.00001)  
    # plt.yscale('symlog', linthresh=.00001)

    # Plot points and labels with appropriate color
    for x, y, label in zip(x_vals, y_vals, keys):
        color = 'orange' if label in subs_pattern_B else 'red'
        label_text= "Pattern B" if label in subs_pattern_B else "Pattern A"
        plt.scatter(x, y, color=color,label=label_text)
        plt.text(x, y, str(label), fontsize=8, ha='left', va='bottom', color='black')
        # plt.xscale('symlog', linthresh=1e-5)  # Adjust linthresh for better visibility
        # plt.yscale('symlog', linthresh=1e-5)
        #plt.text(x, y, label, fontsize=9, ha='right', va='bottom', color=color)
        plt.xlabel("|STFT| diff. between both tasks in left hemi")
        plt.ylabel("|STFT| diff. between both tasks in right hemi")
    plt.savefig(fr"final")
    stop=1
def res_stats(x_vals,y_vals,confidences_multi):
    r_x, p_value_x = spearmanr(np.array(x_vals).flatten(), np.array(list(confidences_multi.values())).flatten())
    r_y, p_value_y = spearmanr(np.array(y_vals).flatten(), np.array(list(confidences_multi.values())).flatten())
    print(r_x, p_value_x)
    print(r_y, p_value_y)

    # X = np.column_stack((x_vals, y_vals))  # shape (n_samples, 2)
    # y = np.array(list(confidences_multi.values()))

    # model = LinearRegression()
    # model.fit(X, y)

    # X = sm.add_constant(X)  # adds intercept
    # model = sm.OLS(y, X).fit()
    # print(model.summary())
    stop=1
def plotting_psds(MI_tf_left_hemi,MI_tf_right_hemi,labels_MI):
    
    unique_labels = np.unique(labels_MI)

    # Prepare scatter plot for individual points
    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        label_text = "LH-MI" if label == 0 else "RH-MI"
        color = plt.cm.viridis((label) / (len(unique_labels) - 1))

        # Plot individual points
        plt.scatter(
            MI_tf_left_hemi[labels_MI == label],
            MI_tf_right_hemi[labels_MI == label],
            c=[color],
            label=label_text,
            s=100
        )

        # Plot average point as a large 'X'
        avg_x = np.mean(MI_tf_left_hemi[labels_MI == label])
        avg_y = np.mean(MI_tf_right_hemi[labels_MI == label])
        plt.scatter(
            avg_x,
            avg_y,
            c=[color],
            s=200,
            marker='X',
            edgecolors='black',
            linewidths=1.5,
            label=f"{label_text} (avg)"
        )

    # Remove tick marks but keep axis labels
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])

    # Labels and title
    plt.xlabel("PSD over Left hemisphere", fontsize=14)
    plt.ylabel("PSD over Right hemisphere", fontsize=14)
    plt.title("MI Distribution", fontsize=16)

    # Legend and grid
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(fr"MI_{sub_mod}.png")
    plt.clf()
def get_dx_dy(MI_tf_left_hemi,MI_tf_right_hemi,labels_MI):
    unique_labels = np.unique(labels_MI)

    for label in unique_labels:

        label_points_left = MI_tf_left_hemi[labels_MI == label]
        avg_left_hemi= np.mean(label_points_left, axis=0)

        label_points_right= MI_tf_right_hemi[labels_MI == label]
        avg_right_hemi= np.mean(label_points_right, axis=0)        
        
        if label==0:
            avgs_left_hemi_LH=avg_left_hemi
            avgs_right_hemi_LH=avg_right_hemi
        else:
            avgs_left_hemi_RH=avg_left_hemi
            avgs_right_hemi_RH=avg_right_hemi            

    dx = avgs_left_hemi_RH - avgs_left_hemi_LH
    dy = avgs_right_hemi_RH - avgs_right_hemi_LH

    return (dx,dy)
def ers_process(case,lbl_id):
    # --- Select MI epochs ---
    mi_mask = labels_all == lbl_id
    mi_full = epochses[mi_mask]      # (n_epochs_MI, n_channels, n_timepoints)

    full_power_series_c3 = []
    full_power_series_c4 = []

    for ep_full in mi_full:
        full_c3 = ep_full[:6,:].mean(axis=0)**2 
        full_c4 = ep_full[6:12,:].mean(axis=0) **2       
        full_power_series_c3.append(full_c3)
        full_power_series_c4.append(full_c4)
    full_power_series_c3 = np.array(full_power_series_c3)
    full_power_series_c4 = np.array(full_power_series_c4)
    full_avg_power_c3 = full_power_series_c3.mean(axis=0)
    full_avg_power_c4 = full_power_series_c4.mean(axis=0)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(full_avg_power_c3)
    axs[0].set_title(f'C3 Normalized Power')
    axs[0].set_ylabel('Normalized Power')
    axs[0].grid(True)
    if case == "lh":
        axs[0].text(0.01, 0.9, "ERS", fontsize=18, transform=axs[0].transAxes)
    else:
        axs[0].text(0.01, -.1, "ERS", fontsize=18, transform=axs[0].transAxes)
    axs[1].plot(full_avg_power_c4)
    axs[1].set_title(f'C4 Normalized Power')
    axs[1].set_ylabel('Normalized Power')
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(fr"{sub_mod}_{case}")
    plt.close()


    step=80
    i_1=80
    i_2=i_1+step//2
    a_max=-10000
    b_max=-10000
    while i_2 < 560: 
        a=((full_avg_power_c3[i_1:i_2].mean()-full_avg_power_c3[:80].mean())/full_avg_power_c3[:80].mean())*100
        b=((full_avg_power_c4[i_1:i_2].mean()-full_avg_power_c4[:80].mean())/full_avg_power_c4[:80].mean())*100

        a_max=max(a,a_max)
        b_max=max(b,b_max)

        i_1+=step//2
        i_2+=step//2

    if case=="lh":
        avgs_c3_lh.append(a_max)
        avgs_c4_lh.append(b_max)
    else:
        avgs_c3_rh.append(a_max)
        avgs_c4_rh.append(b_max)








#Load PhysioNet paths
physionet_paths = [ mne.datasets.eegbci.load_data(id,IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,"/root/mne_data",) for id in range(1, needed_subs + 1)  ]
physionet_paths = np.concatenate(physionet_paths)

# Read EDF files
raws = [mne.io.read_raw_edf(path,preload=True,stim_channel='auto',verbose='WARNING',) for path in physionet_paths]

# Process runs in groups of 3 (each subject)
my_dic_x={}
my_dic_y={}
delta_x_dic={}
delta_y_dic={}
avgs_c3_lh=[]
avgs_c4_lh=[]
avgs_c3_rh=[]
avgs_c4_rh=[]
for beg_global_run_idx in range(0, len(raws), num_runs_sub):

    sub_mod = beg_global_run_idx // num_runs_sub + 1
    labels_MI_l=[]
    x_left_hemi_l=[]
    x_right_hemi_l=[]   
    psd_left_l=[]
    psd_right_l=[] 
    epochs_l=[]
    baseline_epochs_l=[]
    labels_MI_ers=[]
    for local_run_index in range(num_runs_sub): 
        global_run_idx = beg_global_run_idx + local_run_index
        if global_run_idx >= len(raws):
            break  
        raw = raws[global_run_idx]

        epoched,epochs_MI_run=raw_MI_processing(raw)
        x_left_hemi,x_right_hemi,y,psd_left,psd_right=MI_data_generation(epoched)

        labels_MI_l.append(y)
        x_left_hemi_l.append(x_left_hemi) 
        x_right_hemi_l.append(x_right_hemi) 
        psd_left_l.append(psd_left)
        psd_right_l.append(psd_right)
        epochs_l.append(epochs_MI_run) 


    labels_all=np.concatenate(labels_MI_l, axis=0)
    MI_tf_left_hemi=np.concatenate(x_left_hemi_l, axis=0)
    MI_tf_right_hemi=np.concatenate(x_right_hemi_l, axis=0)
    psd_left_ses=np.concatenate(psd_left_l, axis=0)
    psd_right_ses=np.concatenate(psd_right_l, axis=0)
    epochses=np.concatenate(epochs_l, axis=0)


    ers_process("lh",0)
    ers_process("rh",1)  

    # plotting_psds(psd_left_ses,psd_right_ses,labels_all)
    # #plotting_psds(MI_tf_left_hemi,MI_tf_right_hemi,labels_all)
    
    
    # dx,dy=get_dx_dy(MI_tf_left_hemi,MI_tf_right_hemi,labels_all)
    # if sub_mod in subs_taken:
    #     delta_x_dic[sub_mod]=dx
    #     delta_y_dic[sub_mod]=dy

# ############################################################################## Both
# x_vals = [delta_x_dic[k] for k in delta_x_dic]
# y_vals = [delta_y_dic[k] for k in delta_y_dic]  

# labels_stat=[]
# for sub in subs_taken:
#     label_text= 1 if sub in subs_pattern_B else -1
#     labels_stat.append(label_text)
# df = pd.DataFrame({'Column1': x_vals, 'Column2': y_vals, 'Column3': labels_stat})
# df.to_csv('stat_out.csv', index=False)


# res_plotting(x_vals,y_vals,subs_taken,subs_pattern_B)
# res_stats(x_vals,y_vals,confidences_multi)

df = pd.DataFrame({
    "C3_LH": avgs_c3_lh,
    "C4_RH": avgs_c4_rh,
})

# Define groups (adjusted to 0-based indexing)
groups = [
    [55,107,89,28,41,17,5,7,92,1,65,77,68,8,105],        
    [47,27,97,46,3,14,59,60,83,90],            
    [0, 2, 4, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26,
29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 48, 49, 50, 
51, 52, 53, 54, 56, 57, 58, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 
75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 91, 93, 94, 95, 96, 98, 99, 
100, 101, 102, 103, 104, 106, 108],       
]

# Prepare Excel writer
with pd.ExcelWriter("raw_ers.xlsx") as writer:
    start_row = 0
    for i, group in enumerate(groups, 1):
        group_df = df.iloc[group].reset_index(drop=True)
        group_df.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False)
        start_row += len(group_df) + 2  # leave 2-row space between groups



stop=1







