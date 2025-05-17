####################################### Libraries
import numpy as np
import mne
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from collections import Counter
import pandas as pd
from scipy.signal import welch
import statsmodels.api as sm
##################################### Inputs
sfreq = 250 # sampling frequency
sess=[i for i in range(1,19)] # list of sessions (for all 9 subjects). Two sessions per subject so the total is 18.
f_low_rest=1   # low frequency
f_high_rest=20 # high frequency
f_low_MI=8   # low frequency
f_high_MI=12 # high frequency
tmin_rest = 1  # start of time for rest[s]
tmax_rest = 59 # end time for rest [s]
tmin_MI = 1
tmax_MI = 4
############################################################################## Rest
patterns={}
confidences={} # absolute
confidences_multi={} # both +ve and -ve
needed_ratio=1.2

def load_data(ses, data_type):
    my_file = fr"C:\Users\mnaser1\OneDrive - Kennesaw State University\Desktop\PhD-S7\Dissertation\Data\2a2b data\full_2a_data\Data\{ses-1}.mat"
    mat_data = scipy.io.loadmat(my_file)
    if data_type == 'rest': # it's actually 2min, I am taking the first min
        my_data_eeg = np.squeeze(mat_data['data'][0][1][0][0][0][:, 0:22]) # the first 22 channels are EEG
        my_data_eog = np.squeeze(mat_data['data'][0][1][0][0][0][:, 22:25]) # the rest are EOG
    elif data_type == 'mi':
        my_data_eeg = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 0:22])
        my_data_eog = np.squeeze(mat_data['data'][0][run+3][0][0][0][:, 22:25])
    return np.hstack([my_data_eeg, my_data_eog]),mat_data
def create_mne_raw(data):
    numbers = list(range(1, 26))
    ch_names = [str(num) for num in numbers]
    ch_types = ['eeg'] * 22 + ['eog'] * 3
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)
    return raw
def source_computing(raw):
    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {
        str(row['name']): np.array([row['x'], row['y'], row['z']])
        for _, row in df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")

    valid_chs = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['loc'] is not None
        and not np.allclose(ch['loc'][:3], 0)
        and not np.isnan(ch['loc'][:3]).any()
]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)
    #raw.plot_sensors(show_names=True)  # just to verify positions
    return raw
def process_rest_data(raw):
    raw.filter(f_low_rest, f_high_rest, fir_design='firwin') # FIR filtration to keep a range of frequencies
    
    raw=source_computing(raw)

    picks = ["8", "9", "14", "15","2","3",   "11", "12", "17", "18", "5", "6"]     # the channels to consider (refer to data description)
    epoch_length_samples = int((tmax_rest-tmin_rest) * raw.info['sfreq'])
    n_samples = len(raw)

    # Creating a one event for the Rest period
    event_times = np.arange(0, n_samples - epoch_length_samples, epoch_length_samples)
    events = np.column_stack((event_times, np.zeros_like(event_times, dtype=int), np.ones_like(event_times, dtype=int)))
    epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin_rest, tmax=tmax_rest, baseline=None, preload=True, picks=picks)
    
    half_ch=len(picks)//2
    data_rest_left  = np.mean(epochs.get_data()[0][:half_ch, :], axis=0) # the first 4 channels are in the left hemisphere
    data_rest_right  = np.mean(epochs.get_data()[0][half_ch:, :], axis=0) # the last 4 channels are in the right hemisphere
    
    _, _, Zxx_left = stft(data_rest_left, sfreq, nperseg=sfreq) # generating time-frequency map using STFT
    _, _, Zxx_right = stft(data_rest_right, sfreq, nperseg=sfreq) # generating time-frequency map using STFT
    
    Zxx_mag_diff=np.abs(Zxx_left)-np.abs(Zxx_right)

    return Zxx_mag_diff
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
    #"10": s0trong, negative (mainly blue)
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
def plotting_rest_maps(data):
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
        vmin=-0.004,
        vmax=0.004
    )

    # Custom colorbar
    cbar = plt.colorbar(im)
    cbar.set_ticks([-0.004, 0, 0.004])
    cbar.set_ticklabels(["Min (-ve)", "0", "Max (+ve)"])
    cbar.ax.tick_params(labelsize=14)

    # Custom x-axis ticks (e.g., 5 ticks between 0 and 60)
    tick_labels = [0, 15, 30, 45, 60]
    tick_positions = [int(i * (data.shape[1] / 60)) for i in tick_labels]
    plt.xticks(tick_positions, [str(i) for i in tick_labels], fontsize=14)
    plt.yticks(fontsize=14)

    # Axis labels
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Frequency [Hz]", fontsize=18)

    # Optional: Add a title
    plt.title(fr"Rest: |STFT| left hemi - |STFT| right hemi", fontsize=18)

    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(fr"Rest_{ses}.png")
    # if ses==5 or ses==17:
    #     plt.show()
    plt.close()

for ses in sess:
    rest_data, _= load_data(ses, 'rest')
    raw = create_mne_raw(rest_data)
    data_rest_diff_tf_mag = process_rest_data(raw) 
    plotting_rest_maps(data_rest_diff_tf_mag)

    avg_mid_freq=np.mean(data_rest_diff_tf_mag[7:13,:],axis=0)
    avg_below_freq=np.mean(data_rest_diff_tf_mag[2:7,:],axis=0)

    res_down, confidence_down=subject_select(avg_mid_freq,avg_below_freq)
    patterns[ses] = res_down
    confidences[ses] = confidence_down

for k in [k for k, v in patterns.items() if v == "Weak"]: #remove weak
    del patterns[k]
    del confidences[k]

subs_taken=list(patterns.keys())
subs_pattern_B = [k for k, v in patterns.items() if v == "Pattern B"]
confidences_multi = {k: (v if patterns[k] == "Pattern B" else -v) for k, v in confidences.items()}

stop=1


############################################################################## MI
def plotting_psds(data_received, labels_MI):
    unique_labels = np.unique(labels_MI)

    # Prepare scatter plot
    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        label_text = "LL-MI" if label == 1 else "RL-MI"
        color = plt.cm.viridis((label - 1) / (len(unique_labels) - 1))

        # Plot individual points
        plt.scatter(
            data_received[labels_MI == label, 0],
            data_received[labels_MI == label, 1],
            c=[color],
            label=label_text,
            s=100
        )

        # Plot average point as a large 'X'
        avg_x = np.mean(data_received[labels_MI == label, 0])
        avg_y = np.mean(data_received[labels_MI == label, 1])
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
    plt.xlabel(r"$\mathrm{PSD}_{\mathrm{MI,\ LH}}$", fontsize=20)
    plt.ylabel(r"$\mathrm{PSD}_{\mathrm{MI,\ RH}}$", fontsize=20)
    plt.title("MI Distribution", fontsize=18)

    # Legend and grid
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(fr"MI_{ses}.png")
    if ses==17:
        plt.show()
    plt.clf()
def get_dx_dy(data_received, labels_MI):
    unique_labels = np.unique(labels_MI)

    for label in unique_labels:

        label_points = data_received[labels_MI == label]
        
        avg_left_hemi, avg_right_hemi = np.mean(label_points, axis=0)
        
        if label==1:
            avgs_left_hemi_LH=avg_left_hemi
            avgs_right_hemi_LH=avg_right_hemi
        else:
            avgs_left_hemi_RH=avg_left_hemi
            avgs_right_hemi_RH=avg_right_hemi            

    dx = avgs_left_hemi_RH - avgs_left_hemi_LH
    dy = avgs_right_hemi_RH - avgs_right_hemi_LH

    return (dx,dy)
def process_mi_data(raw, mat_data):
    raw.filter(f_low_MI, f_high_MI, fir_design='firwin') # FIR filtration to keep a range of frequencies

    df = pd.read_csv(fr"25montage.csv", header=None, names=["name", "x", "y", "z"])
    ch_pos = {
        str(row['name']): np.array([row['x'], row['y'], row['z']])
        for _, row in df.iterrows()
    }

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing="warn")


    valid_chs = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['loc'] is not None
        and not np.allclose(ch['loc'][:3], 0)
        and not np.isnan(ch['loc'][:3]).any()
]

    raw = raw.copy().pick_channels(valid_chs)

    raw = mne.preprocessing.compute_current_source_density(raw)



    events = np.squeeze(mat_data['data'][0][run+3][0][0][2]) # only the first run of each session is taken (total number of trials is 48, only left and right hand considered so 24)
    event_indices = np.squeeze(mat_data['data'][0][run+3][0][0][1])
    mne_events = np.column_stack((event_indices, np.zeros_like(event_indices), events))
    picks = ["8", "9", "14", "15","2","3",   "11", "12", "17", "18", "5", "6"] 

    event_id_MI = dict({'769': 1, '770': 2})
    full_epochs=mne.Epochs(raw, mne_events, event_id_MI, 0, 4, proj=True,  baseline=None, preload=True, picks=picks)
    labels_MI = full_epochs.events[:, -1]
    data_MI_original = full_epochs.get_data()

    _, _, Zxx = stft(data_MI_original, 250, nperseg=250) # generating time-frequency map using STFT
    MI_tf=np.abs(Zxx)
    f, psd_run = welch(data_MI_original, fs=sfreq)

    return (labels_MI,MI_tf,psd_run,full_epochs)
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
        plt.xlabel("PSD diff. between both tasks in left hemi")
        plt.ylabel("PSD diff. between both tasks in right hemi")
    plt.savefig(fr"final")
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
def ers_process(case,lbl_id):
    # --- Select MI epochs ---
    mi_mask = labels_MI == lbl_id
    mi_full = full_epochs[mi_mask]      # (n_epochs_MI, n_channels, n_timepoints)

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
    plt.savefig(fr"{ses}_{case}")
    plt.close()

    step=125
    i_1=125
    i_2=i_1+step//2
    a_max=-10000
    b_max=-10000
    while i_2 < 875: 
        a=((full_avg_power_c3[i_1:i_2].mean()-full_avg_power_c3[:125].mean())/full_avg_power_c3[:125].mean())*100
        b=((full_avg_power_c4[i_1:i_2].mean()-full_avg_power_c4[:125].mean())/full_avg_power_c4[:125].mean())*100

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








delta_x_dic={}
delta_y_dic={}
avgs_c3_lh=[]
avgs_c4_lh=[]
avgs_c3_rh=[]
avgs_c4_rh=[]

for ses in sess:
    data_MI_l=[]
    labels_MI_l=[]
    MI_tf_l=[]
    psd_l=[]
    epochs_l=[]
    baseline_epochs_l=[]
    full_epochs_run_l=[]
    for run in range(0,3):
        mi_data, mat_data = load_data(ses, 'mi')
        raw = create_mne_raw(mi_data)

        labels_MI_run,MI_tf_run,psd_run,full_epochs_run = process_mi_data(raw, mat_data)
        labels_MI_l.append(labels_MI_run)
        MI_tf_l.append(MI_tf_run) 
        psd_l.append(psd_run) 
        full_epochs_run_l.append(full_epochs_run)

    labels_MI=np.concatenate(labels_MI_l, axis=0)
    MI_tf=np.concatenate(MI_tf_l, axis=0)
    psds=np.concatenate(psd_l, axis=0)
    full_epochs=np.concatenate(full_epochs_run_l, axis=0)

    ###############################################################
    baseline_length = 1 #[s]
    baseline_samples = int(baseline_length*sfreq)
    
    ers_process("lh",1)
    ers_process("rh",2)    

    data_MI_tf_abs_avg_freq=np.mean(np.squeeze(MI_tf),axis=2) 
    psds_avg_freq=np.mean(np.squeeze(psds),axis=2) 
    data_MI_tf_abs_avg_freq_time=np.mean(np.squeeze(data_MI_tf_abs_avg_freq),axis=2) 

    data_MI_tf_abs_avg_freq_time_left_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, :6], axis=1) 
    data_MI_tf_abs_avg_freq_time_right_hemi=np.mean(data_MI_tf_abs_avg_freq_time[:, 6:], axis=1) 
    psds_avg_freq_left_hemi = np.mean(psds_avg_freq[:, :6], axis=1) 
    psds_avg_freq_right_hemi = np.mean(psds_avg_freq[:, 6:], axis=1) 
    data_mi_stacked_tf = np.vstack((data_MI_tf_abs_avg_freq_time_left_hemi, data_MI_tf_abs_avg_freq_time_right_hemi)).T  # Combine along the second axis
    psds_stacked =  np.vstack((psds_avg_freq_left_hemi, psds_avg_freq_right_hemi)).T 

    plotting_psds(psds_stacked,labels_MI) 
    #plotting_psds(data_mi_stacked_tf,labels_MI)

    dx,dy=get_dx_dy(psds_stacked,labels_MI) # dx and dy are the difference between the two MI tasks for each hemisphere (x: left hemi, y: right hemi)
    #dx,dy=get_dx_dy(data_mi_stacked_tf,labels_MI) # dx and dy are the difference between the two MI tasks for each hemisphere (x: left hemi, y: right hemi)


    if ses in subs_taken:
        delta_x_dic[ses]=dx
        delta_y_dic[ses]=dy

############################################################################## Both
x_vals = [delta_x_dic[k] for k in delta_x_dic]
y_vals = [delta_y_dic[k] for k in delta_y_dic]  

labels_stat=[]
for sub in subs_taken:
    label_text= 1 if sub in subs_pattern_B else -1
    labels_stat.append(label_text)
df = pd.DataFrame({'Column1': x_vals, 'Column2': y_vals, 'Column3': labels_stat})
df.to_csv('stat_out.csv', index=False)

res_plotting(x_vals,y_vals,subs_taken,subs_pattern_B)
res_stats(x_vals,y_vals,confidences_multi)

df = pd.DataFrame({
    "C3_LH": avgs_c3_lh,
    "C4_RH": avgs_c4_rh,
})

# Define groups (adjusted to 0-based indexing)
groups = [
    [6, 7, 10, 11, 16],        # Group 1: indices 7,8,11,12,17,18
    [4, 8, 5, 14, 15,17],             # Group 2: indices 5,9,6,15,16
    [0, 1, 2, 3, 9, 12, 13],       # Group 3: indices 1,2,3,4,10,13,14
]

# Prepare Excel writer
with pd.ExcelWriter("raw_ers.xlsx") as writer:
    start_row = 0
    for i, group in enumerate(groups, 1):
        group_df = df.iloc[group].reset_index(drop=True)
        group_df.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False)
        start_row += len(group_df) + 2  # leave 2-row space between groups



stop=1
