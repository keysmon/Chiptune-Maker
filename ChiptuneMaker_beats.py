import os
import sys
import IPython.display as ipd
import numpy as np
import soundfile as sf
import shutil
import pyrubberband as pyrb
import scipy.io.wavfile as wav
import crepe
from scipy.interpolate import interp1d
import warnings
import PySynth_custom.pysynth as pysynth
import PySynth_custom.pysynth_b as pysynth_b
import PySynth_custom.pysynth_c as pysynth_c
import PySynth_custom.pysynth_d as pysynth_d
import parselmouth
import librosa
from spleeter.separator import Separator
import statistics



def load_wav(fname):
    # srate, audio = wav.read(fname)
    # audio = audio.astype(np.float32) / 32767.0
    # audio = (0.9 / np.max(audio)) * audio
    audio,srate = librosa.load(fname)
    # convert to mono
    if (len(audio.shape) == 2):
        audio = (audio[:, 0] + audio[:, 1]) / 2
    return (audio,srate)


# Function to separate instrument stems in a file using Spleeter
# Input: audio file, number of stems {2,4,5}
# Output: None in program, creates desired number of stem wav files in specified path
def stem_separate(song_path, num_stems=4):
    try:
        shutil.move(os.path.join('temp_data', 'pretrained_models'), os.getcwd())             #move spleeter helper folder to main folder
    except:
        pass

    separator = Separator('spleeter:' + str(num_stems) + 'stems')
    separator.separate_to_file(song_path, os.path.join("temp_data", "spleeter_output"))

    shutil.move('pretrained_models', 'temp_data')            #move it back

    return


# Function to break stem into frames of 16th note length and calc the note being played in that frame
def stem_freq_beat_track(stem_path,sr,onsets):
    audio,sr = load_wav(stem_path)
    audio_p = parselmouth.Sound(stem_path)
    sr_p = audio_p.sampling_frequency

    #first arr is time, second midi
    notes = [[],[]]
    for i in range(len(onsets)-1):
        frame = audio[onsets[i]:onsets[i+1]]
        frame_p = audio_p[onsets[i]:onsets[i+1]]
        #divide in 4 parts (into 16th notes)
        hop_size = round(len(frame) / 4)
        offsets = np.arange(0,len(frame),hop_size)
        for j,o in enumerate(offsets):
            subframe = frame[o:o+hop_size]
            subframe_p = frame_p[o:o+hop_size]
            if 'bass' in stem_path:
                #crepe works good for bass
                time,sframe_freqs,conf,act = crepe.predict(subframe,sr,viterbi=True,verbose=False)
            else:
                #pyin
                # sframe_freqs,flag,conf = np.nan_to_num(librosa.pyin(y=subframe,fmin=27.5,fmax=4100.0,sr=sr))

                #crepe
                time,sframe_freqs,conf,act = crepe.predict(subframe,sr,viterbi=True,verbose=False)

                # #parselmouth
                # sd = frame_p.extract_part(from_time=o/sr_p, to_time=(o+hop_size)/sr_p, preserve_times=True)
                # sframe_freqs = [sd.to_pitch(pitch_floor=27.50)]  # pitch floor is A0
                # conf = [1.0]

            # keep only frequencies with high confidence
            sframe_midi = [round(librosa.hz_to_midi(x)) if x > 27.0 and y > 0.5 else 0 for x, y in
                          list(zip(sframe_freqs, conf))]
            #take most common note in subframe
            note_midi = statistics.mode(sframe_midi)
            notes[1].append(note_midi)
            notes[0].append((onsets[i]+o)/sr)

    return notes

# Function to convert time/midi information to a pysynth script
def pysynth_script(midi_time,bpm):
    # maybe alter to accept notes longer than whole
    def closest_note(duration, notes):
        if duration == 0:
            return notes

        durations = {1: 16,  # 16th
                     2: 8,  # eighth
                     #3: -8,  # dotted eighth
                     4: 4,  # quarter note
                     #6: -4,  # dotted quarter
                     8: 2,
                     #12: -2,
                     16: 1  # whole note
                     }

        # take closest without going over
        d1 = np.array(list(durations.keys()))
        d2 = d1 - duration
        filter = d2 <= 0
        closest = np.max(d1[filter])
        remainder = duration - closest
        notes.append(durations[closest])
        return closest_note(remainder, notes)

        return notes

    s = []
    prev_note = midi_time[1][0]
    dur = 0    #in terms of 16th notes
    for i in range(1,len(midi_time[0])):
        time = midi_time[0][i]
        note = midi_time[1][i]

        # if prev_note == 0:
        #     note_letter = 'r'
        # else:
        #     note_letter = librosa.midi_to_note(prev_note)
        #     note_letter = note_letter[0].lower()+note_letter[1:]
        #     if len(note_letter) == 3:
        #         if 'b' not in note_letter:
        #             note_letter = note_letter[0]+'#'+note_letter[2]
        # s.append((note_letter,16))

        dur += 1
        # new note detected
        if note != prev_note:
            # get the duration of the note in PySynth's format
            notes = []
            notes = closest_note(dur,notes)

            #get note letter eg. C5
            if prev_note == 0:
                note_letter = 'r'
            else:
                note_letter = librosa.midi_to_note(prev_note)
                note_letter = note_letter[0].lower()+note_letter[1:]
                if len(note_letter) == 3:
                    if 'b' not in note_letter:
                        note_letter = note_letter[0]+'#'+note_letter[2]

            # write to script
            for note_dur in notes:
                s.append((note_letter,note_dur))

            # update for next iteration
            dur = 0
        prev_note = note
    return s

# Function to create chiptune drum track
def drum_synth(stem_path,song_name, bpm_source):
    # Function to remove all but kick and snare sounds from drum track
    def drum_scrub(drum_audio, sr, bpm, amp=0.6):

        drum_scrubbed = np.zeros(len(drum_audio))
        max_note = round((bpm/60/2/2/2/2)*sr)  # 64th note

        j=0
        while j<len(drum_audio):
            if np.abs(drum_audio[j]) > float(amp):
                drum_scrubbed[j:j+max_note] = drum_audio[j:j+max_note]
                j += max_note
            else: j+=1
        return drum_scrubbed

    drum_audio,sr = librosa.load(stem_path)
    scrubbed = drum_scrub(drum_audio,sr,bpm_source,0.5)
    drum_sound,_ = librosa.load(os.path.join('instrument_sounds','synth_drum.wav'))

    drum_synthed = np.zeros(len(drum_audio))
    detected = False
    start = 0

    #minimum drum beat is between 16th and 32th note
    min_beat = bpm_source/60/2/2/1.95

    for i,amp in enumerate(scrubbed):
        if amp > 0.0 and not detected:
            detected = True
            start = i
        if amp == 0.0 and detected and i > start + int(min_beat*sr):
            detected = False
            length = min(len(drum_sound), len(drum_synthed) - i)
            drum_synthed[start:start + length] += drum_sound[:length]

    path = os.path.join('temp_data','chiptune_stems',song_name,'drums.wav')
    sf.write(path,drum_synthed,samplerate=sr)
    return

# function to mix together chiptune stems
def mixer(song_name,onsets):
    chiptune_directory = os.path.join('temp_data','chiptune_stems',song_name)
    tracks = []
    length = 0
    for stem_name in os.listdir(chiptune_directory):
        stem_path = os.path.join(chiptune_directory,stem_name)

        # load stem
        stem_audio, sr = librosa.load(stem_path)

        #shift pitch detected stems
        if stem_name != 'drums.wav':
            stem_audio = np.concatenate((np.zeros(onsets[0]),stem_audio))
        else:
            length = len(stem_audio)

        tracks.append(stem_audio)

    #pad
    for i,t in enumerate(tracks):
        if len(t)<length:
            pad = length-len(t)
            tracks[i] = np.concatenate((t,np.zeros(pad)))

    #mix stems
    mixed_track = np.vstack(tracks)
    mixed_track = np.sum(mixed_track,axis=0)
    return mixed_track

# Fuction that hard codes the stems. Used for combining chiptune stems with original
def fugasie_mix(song_name,onsets):
    chiptune_directory = os.path.join('temp_data', 'chiptune_stems', song_name)

    drums = librosa.load(chiptune_directory+'/drums.wav')[0]

    tracks = [
    drums,
    librosa.load(chiptune_directory+'/bass.wav')[0],
    librosa.load(os.path.join('temp_data','spleeter_output',song_name,'vocals.wav'))[0],
    librosa.load(os.path.join('temp_data','spleeter_output',song_name,'other.wav'))[0]
    ]

    length = len(drums)

    # pad
    for i, t in enumerate(tracks):
        if len(t) < length:
            pad = length - len(t)
            tracks[i] = np.concatenate((t, np.zeros(pad)))

    # mix stems
    mixed_track = np.vstack(tracks)
    mixed_track = np.sum(mixed_track, axis=0)
    return mixed_track


# MAIN------------------------------------------------------------
def main(audio_file="rock.00031.wav", num_of_stem=4):
    # set up paths and variables needed
    input_song_path = os.path.join("input_songs", audio_file)  # path to source song
    input_song_name = audio_file[:-4]  # source song name

    os.makedirs('temp_data', exist_ok=True)  # make "temp_data" folder to hold files created
    os.makedirs('output', exist_ok=True)  # make "output" folder

    # load source song
    song_audio, sr = librosa.load(input_song_path)

    stem_directory = os.path.join('temp_data', 'spleeter_output', input_song_name)
    output_directory = os.path.join('temp_data','chiptune_stems',input_song_name)

    # BEGIN-----------------------------------------------------
    # PHASE 1: Separate song in to stems
    stem_separate(song_path=input_song_path, num_stems=num_of_stem)

    # get bpm of song
    drum_audio,_ = librosa.load(os.path.join(stem_directory,'drums.wav'))
    potential_bpms = (librosa.beat.beat_track(song_audio,sr)[0],
                      librosa.beat.beat_track(drum_audio,sr)[0],
                      librosa.beat.tempo(song_audio,sr)[0],
                      librosa.beat.tempo(drum_audio,sr)[0])
    potential_bpms = [round(x,2) for x in potential_bpms]
    bpm = statistics.mode(potential_bpms)

    #get beat onsets
    onsets = librosa.beat.beat_track(song_audio,units='samples')[1]

    for stem_name in os.listdir(stem_directory):
        print(stem_name)
        stem_path = os.path.join(stem_directory, stem_name)
        output_path = os.path.join(output_directory,stem_name)

        # chiptune drums
        if stem_name == 'drums.wav':
            drum_synth(stem_path,input_song_name,bpm)
        else:
            #get binned frequencies
            stem_freqs = stem_freq_beat_track(stem_path,sr,onsets)

            # convert freqs to PySynth script
            stem_py_script = pysynth_script(stem_freqs,bpm)

            # write chiptune to wav file
            os.makedirs(output_directory,exist_ok=True)
            # CHANGE PYSYNTH INSTRUMENT HERE..................
            pysynth_d.make_wav(stem_py_script, fn=output_path, bpm=bpm)


    # Mix chiptune stems together
    final_track = mixer(input_song_name,onsets)

    # Output song
    sf.write(os.path.join('output', input_song_name + '_chiptune.wav'), final_track, samplerate=sr)

    return



if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print("Error: Invalid Args!")
