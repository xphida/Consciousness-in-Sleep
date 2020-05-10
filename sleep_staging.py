# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:38:12 2020

@author: xphid
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


#%%

subject_1, subject_2 = 0, 1

[subject_1_files, subject_2_files] = fetch_data(subjects=[subject_1, subject_2], recording=[1])

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

raw_train = mne.io.read_raw_edf(subject_1_files[0])
annot_train = mne.read_annotations(subject_1_files[1])

raw_train.set_annotations(annot_train, emit_warning=False)
raw_train.set_channel_types(mapping)

# plot some data
raw_train.plot(duration=60, scalings='auto')


#%%

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4      (dictionary, key : value)
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# plot events
mne.viz.plot_events(events_train, event_id=event_id,
                    sfreq=raw_train.info['sfreq'])

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%%

tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                          event_id=event_id, tmin=0., tmax=tmax, baseline=None)

print(epochs_train)

#%%

raw_test = mne.io.read_raw_edf(subject_2_files[0])
annot_test = mne.read_annotations(subject_2_files[1])
raw_test.set_annotations(annot_test, emit_warning=False)
raw_test.set_channel_types(mapping)
events_test, _ = mne.events_from_annotations(
    raw_test, event_id=annotation_desc_2_event_id, chunk_duration=30.)
epochs_test = mne.Epochs(raw=raw_test, events=events_test, event_id=event_id,
                         tmin=0., tmax=tmax, baseline=None)

print(epochs_test)

#%%

# visualize subject_1 vs. subject_2 PSD by sleep stage.
fig, (ax1, ax2) = plt.subplots(ncols=2)

# iterate over the subjects
stages = sorted(event_id.keys())
for ax, title, epochs in zip([ax1, ax2],
                             ['subject 1', 'subject 2'],
                             [epochs_train, epochs_test]):

    for stage, color in zip(stages, stage_colors):
        epochs[stage].plot_psd(area_mode=None, color=color, ax=ax,
                               fmin=0.1, fmax=20., show=False,
                               average=True, spatial_colors=False)
    ax.set(title=title, xlabel='Frequency (Hz)')
ax2.set(ylabel='µV^2/Hz (dB)')
ax2.legend(ax2.lines[2::3], stages)
plt.show()

#%%

#create a function that compare PSD's EEG to predict sleep's phases

def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}
    
    """ 
    Compute the power spectral density (PSD) using Welch’s method:
    Calculates periodograms for a sliding window over the time dimension,
    then averages them together for each channel/epoch.
    
    epochs : instance of Epochs 
        The data for PSD calculation.
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    picks : Channels to include. 
    Slices and lists of integers will be interpreted as channel indices. 
    In lists, channel type strings like ['eeg']
    return :
    -------
    
    psds :  power spectral densities
    freqs : frequencies
        
       
ALTRA SCRITTURA CON MNE

n_fft = int(raw.info['sfreq'] / resolution)
psd, freqs = mne.time_frequency.psd_welch(raw, picks=picks,
                                              n_fft=n_fft, fmin=1.,
                                              fmax=30.)
"""


    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

#%%

#MULTICLASS CLASSIFICATION  (RANDOM FOREST)

"""
Scikit-learn pipeline composes an estimator as a sequence of transforms and a final estimator,
 while the FunctionTransformer converts a python function in an estimator compatible object. 
 In this manner we can create scikit-learn estimator that takes mne.Epochs thanks 
 to eeg_power_band function we just created.
"""

"""
Sklearn Functions:
make_pipeline = Construct a Pipeline from the given estimators. 
This is a shorthand for the Pipeline constructor.
Composes an estimator as a sequence of transforms and a final estimator.

Functiontrasformer = converts a python function in an estimator compatible object.
Constructs a transformer from an arbitrary callable.
A FunctionTransformer forwards its X (and optionally y) arguments to a 
user-defined function or function object and returns the result of this function.
This is useful for stateless transformations such as taking the log of frequencies,
doing custom scaling, etc. 
"""

#using functiontrasformer I can use the functions in the pipeline to transform the dataframe (making a 0custom scaling)
pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     RandomForestClassifier(n_estimators=100, random_state=42))


# Train
y_train = epochs_train.events[:, 2]      
pipe.fit(epochs_train, y_train)    # fit : adjusts weights according to data values so that better accuracy can be achieved.
# Test
y_pred = pipe.predict(epochs_test) # predict : After training, the model can be used for predictions, using . predict() method call. 

# Assess the results                      
y_test = epochs_test.events[:, 2]
acc = accuracy_score(y_test, y_pred)    

print("Accuracy score: {}".format(acc))          # it assess if I can predict subject 2's sleeping stages based on subject 1’s data.

#%%

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# classification report  (precision , recall , f1 score, support)
"""
Precision:  is the ratio of correctly predicted positive observations to the total predicted positive observations.
             Precision = TP/TP+FP
Recall:  (Sensitivity) -  is the ratio of correctly predicted positive observations to the all observations in actual class.
             Recall = TP/TP+FN
F1 score: F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. 
          The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.
             F1 Score = 2*(Recall * Precision) / (Recall + Precision)
Support: is the number of samples of the true response that lie in that class.
"""
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

#%%

# create a dataframe 2802 rows x 4 column  ( sleep stages , epochs , start time , stop time )

import numpy as np
import pandas as pd 


df = pd.DataFrame ({'sleep_stages': (y_pred)}) # potevo farlo tutto così
df['epochs'] = (np.arange(start= 0 , stop = ( np.size (y_test)))) +1     # array of number of epochs
df['start_time'] = ((np.arange(start= 0 , stop = ( np.size (y_test)))) ) * 3000.0   # start time (s)  30 s = 1 epoch   
df['stop_time'] = (((np.arange(start= 0 , stop = ( np.size (y_test)))) ) * 3000.0) +3000 
print(df)

#%%

# create a dataframe that contains just NREM 3-4 epochs  (135 rows x 3 columns)

print(event_id)

df_nrem4 =df[df.sleep_stages == 4]
print(df_nrem4)

#%%

""" eeg time = sampled 100 Hz (100 samples for second)  """

just_time= df_nrem4.drop(['sleep_stages', 'epochs'], axis=1)    # oppure mantengo il numero dell'epoca

print(just_time)


# lo volevo trasformare in un array

#epoch_time = just_time.to_numpy()      # 135 * 2    [[][][]]

#epoch_time = just_time.to_numpy().T   #trasposto  135 * 2   [[      ]]
#print(epoch_time)
 
#%%

#rappresento l'eeg in un dataset pandas

dfeeg_data= raw_test.to_data_frame(picks= 'eeg')  # in qusto caso faccio con pandas perchè può selezionare solo i canali eeg altrimenti avrei usato direttamente numpy
print(dfeeg_data)

#%%

# slincing in epoche del segnale raw  , cioè taglio l'eeg in slices corrispondenti a al dataset just__time


""" provo a cambiare approccio e tagliare non il tempo ma le epoche che mi creo da sola
tagliando il tracciato eeg ((il dataset di pandas : dfeeg_data )) in slices da  3000  = 30 s """

"""
maxtimeeeg = int (dfeeg_data.time.index[-1])

#epochduration = np.linspace 
#print(epochduration)

matriceprova= []     #lista vuota



QUESTA VA BENE: 

while (maxtimeeeg//3000):
    for maxtimeeeg in range (maxtimeeeg//3000):
      matriceprova.append(dfeeg_data[0:(3000 * (maxtimeeeg//3000))]) 
                                     
matriceprova=np.asarray(matriceprova)

print(matriceprova)

"""