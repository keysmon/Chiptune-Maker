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


with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=DeprecationWarning)
    import librosa
    from spleeter.separator import Separator


BASS_SCALE = 3*12 # octaves

def load_wav(fname):
    srate, audio = wav.read(fname)
    audio = audio.astype(np.float32) / 32767.0
    audio = (0.9 / np.max(audio)) * audio
    # convert to mono
    if (len(audio.shape) == 2):
        audio = (audio[:, 0] + audio[:, 1]) / 2
    return (audio,srate)


# Function to separate instrument stems in a file using Spleeter
# Input: audio file, number of stems {2,4,5}
# Output: None in program, creates desired number of stem wav files in specified path
def stem_separate(song_path, num_stems=5):
    try:
        shutil.move(os.path.join('temp_data', 'pretrained_models'), os.getcwd())             #move spleeter helper folder to main folder
    except:
        pass

    separator = Separator('spleeter:' + str(num_stems) + 'stems')
    separator.separate_to_file(song_path, os.path.join("temp_data", "spleeter_output"))

    shutil.move('pretrained_models', 'temp_data')            #move it back

    return


def pysynth_script(freq_times,bpm):
    # maybe alter to accept notes longer than whole
    def closest_note(q_length, note_dur):
        # durations used by pysynth
        ratios = (1,  # whole
                  2,  # half
                  4,  # quater
                  8,  # eighth
                  16)  # sixteenth
        note_lengths = np.array([q_length * 4,  # whole note
                                 q_length * 2,  # half note
                                 q_length * 1,  # quarter note
                                 q_length * 0.5,  # eighth note
                                 q_length * 0.25])  # 16th
        best_note = np.argmin(np.absolute(note_lengths - note_dur))
        return ratios[best_note]

    s = []
    # quarter note length in seconds
    q_length = 60/bpm
    min_note = q_length/2/2 #16th note

    # convert to midi
    midi_time = [[],[]]
    for i,f in enumerate(freq_times[1]):
        midi_time[0].append(freq_times[0][i])
        midi_time[1].append(round(librosa.hz_to_midi(freq_times[1][i])))

    prev_note = midi_time[1][0]
    note_start = 0
    for i in range(1,len(midi_time[0])):
        time = midi_time[0][i]
        note = midi_time[1][i]

        # new note detected
        if note != prev_note:
            note_dur = time-note_start  # duration of note in seconds

            # only write notes that are long enough
            if note_dur >= min_note:
                print(note)
                # get the duration of the note in PySynth's format
                dur = closest_note(q_length,note_dur)

                #get note letter eg. C5
                note_letter = librosa.midi_to_note(note)

                note_letter = note_letter[0].lower()+note_letter[1:]
                if len(note_letter) == 3:
                    if 'b' not in note_letter:
                        note_letter = note_letter[0]+'#'+note_letter[2]

                # write to script
                s.append((note_letter,dur))

            # update for next iteration
            note_start = time
        prev_note = note

    return s


# Function to remove excess sound from stem. Gets fundamental frequency
# Input: stem path
# Return: None (original stems replaced)
def stem_process(song_name, bpm, sr):
    hop_size = 1024

    stem_directory = os.path.join('temp_data', 'spleeter_output', song_name)  # path to .wav stem files
    os.makedirs(os.path.join('temp_data', 'processed_stems'), exist_ok=True)             # create folder for processed stems files
    os.makedirs(os.path.join('temp_data', 'processed_stems',song_name), exist_ok=True)

    output_path = os.path.join('temp_data', 'processed_stems',song_name)

    for wav_stem in os.listdir(stem_directory):
        stem_path = os.path.join(stem_directory,wav_stem)
        # load stem
        stem_audio,sr = load_wav(stem_path)
        # stem_audio,sr = librosa.load(stem_path)

        if wav_stem == 'drums.wav':
            processed_stem = drum_scrub(stem_audio, bpm, sr, amp=0.5)

        elif wav_stem == 'other.wav': continue

        else:
            # shift the bass up by BASS_SHIFT
            if wav_stem == 'bass.wav':
                stem_audio = pyrb.pitch_shift(stem_audio, sr, BASS_SCALE)

            # if wav_stem == 'vocals.wav':
            #     other_audio,sr = load_wav(os.path.join(stem_directory,'other.wav'))
            #     stem_audio = np.vstack([stem_audio,other_audio])
            #     stem_audio = np.sum(stem_audio, axis=0)

            # predict frequency
            time, stem_frequency, confidence, activation = crepe.predict(stem_audio, sr, viterbi=True)
            # stem_frequency = np.nan_to_num(librosa.pyin(stem_audio, fmin=20, fmax=3000, sr=sr)[0])

            print(wav_stem)
            stem_f = []
            for t,f,c,a in list(zip(time,stem_frequency,confidence,activation)):
                if c > 0.5:
                    stem_f.append(round(f))

                if f>1000.0:
                    print(t,f,c)

            with open(wav_stem[:-4]+'.txt','w') as o:
                for i,t in enumerate(time):
                    o.write(str(t)+',')
                for i,f in enumerate(stem_frequency):
                    o.write(str(f)+',')


            # sonify into audio
            processed_stem = sonify(stem_f, sr, hop_size)

            # convert to original speed
            processed_stem = librosa.effects.time_stretch(processed_stem, 2)

        # overwrite original stem
        sf.write(os.path.join(output_path,wav_stem), processed_stem, samplerate=sr)

    return

# Function to sonify a pitch track into a sine wave
# Taken from "csc475-575-Spring2022-assignment6.ipynb"
def sonify(pitch_track, srate, hop_size):

    times = np.arange(0.0, float(hop_size * len(pitch_track)) / srate,
                      float(hop_size) / srate)

    # sample locations in time (seconds)
    sample_times = np.linspace(0, np.max(times), int(np.max(times)*srate-1))

    # create linear interpolators for frequencies and amplitudes
    # so that we have a frequency and amplitude value for
    # every sample
    freq_interpolator = interp1d(times,pitch_track)

    # use the interpolators to calculate per sample frequency and
    # ampitude values
    sample_freqs = freq_interpolator(sample_times)

    # create audio signal
    audio = np.zeros(len(sample_times));
    T = 1.0 / srate
    phase = 0.0

    # update phase according to the sample frequencies
    for i in range(1, len(audio)):
        audio[i] =  np.sin(phase)
        phase = phase + (2*np.pi*T*sample_freqs[i])

    return audio

# Function that converts audio files to midi files.
# Input: stem wav file from spleeter
# Output: midi files
def audio2midi(song_name):

    stem_directory = os.path.join('temp_data', 'processed_stems', song_name)  # path to .wav stem files
    os.makedirs(os.path.join('temp_data', 'midi'), exist_ok=True)             # create folder for midi files
    os.makedirs(os.path.join('temp_data', 'midi', song_name), exist_ok=True)  # create fold for source song
    audio2midi_path = os.path.join("audio_to_midi", "audio2midi.py")    # path to audio2midi.py tool

    for stem_path in os.listdir(stem_directory):

        #skip drums
        if stem_path == 'drums.wav': continue

        midi_path = os.path.join('temp_data', "midi", song_name, stem_path[:-4] + ".mid")  # create blank .mid
        stem_path = os.path.join(stem_directory,stem_path)
        output_midi = open(midi_path, 'w')

        # using the command line tool for audio_to_midi.
        try:
            os.system("python " + audio2midi_path + " {} {}".format(stem_path, midi_path))
        except:
            try:
                os.system("python3 " + audio2midi_path + " {} {}".format(stem_path, midi_path))
            except:
                print("Error with audio2midi package.")
                sys.exit()

        output_midi.close()
    return


# Function that synthesizes midi files into chiptune
# Input: name of the input song
# Returns: list of paths to chiptune stem wav files
def PySynth_synthisize(song_name, bpm_source):
    midi_directory = os.path.join('temp_data', 'midi', song_name)                       # midi file directory
    pysynth_path = os.path.join('PySynth','readmidi.py')                                # pysynth path
    os.makedirs(os.path.join('temp_data', 'chiptune_stems', song_name), exist_ok=True)  # create folder for chiptune stems
    tracks = []
    instruments = {      # instruments for each stem. '' is default --syn_a
        'vocals': '',
        'piano': '--syn_b',
        'bass': '--syn_e',
        'other': '--syn_b'}
    for midi_file in os.listdir(midi_directory):
        os.system("python " + pysynth_path + " {} {} {} {}".format(os.path.join(midi_directory, midi_file), 1,
                                                                   os.path.join('temp_data', 'chiptune_stems',
                                                                                song_name,
                                                                                midi_file[:-4] + '.wav'),
                                                                   instruments[midi_file[:-4]]))
        tracks.append(os.path.join('temp_data', 'chiptune_stems', song_name, midi_file[:-4] + '.wav'))


    tracks.append(drum_synth(song_name, bpm_source))

    return tracks

# Function to create chiptune drum track
def drum_synth(song_name, bpm_source):
    drum_stem_path = os.path.join('temp_data','processed_stems',song_name,'drums.wav')
    drum_audio,sr = librosa.load(drum_stem_path)
    drum_sound,_ = librosa.load(os.path.join('instrument_sounds','synth_drum.wav'))
    drum_synthed = np.zeros(len(drum_audio))
    detected = False
    start = 0

    #minimum drum beat is between 16th and 32th note
    min_beat = bpm_source/60/2/2/1.95

    for i,amp in enumerate(drum_audio):
        if amp > 0.0 and not detected:
            detected = True
            start = i
        if amp == 0.0 and detected and i > start + int(min_beat*sr):
            detected = False
            length = min(len(drum_sound), len(drum_synthed) - i)
            drum_synthed[start:start + length] += drum_sound[:length]

    path = os.path.join('temp_data','chiptune_stems',song_name,'drums.wav')
    sf.write(path,drum_synthed,samplerate=sr)

    return path


# Function to synthesize midi files into chiptune using Matthew's implementation
# Input name of source song
# Returns list of
def chiptune_synth(song_name, sr):
    midi_directory = os.path.join('temp_data', 'midi', song_name)  # midi file directory
    chiptune_directory = os.path.join('temp_data', 'chiptune_stems', song_name)
    os.makedirs(os.path.join(chiptune_directory), exist_ok=True)  # create folder for chiptune stems
    instruments = {      # instruments for each stem
        'drums.mid': 'square',
        'vocals.mid': 'sin',
        'piano.mid': 'square',
        'bass.mid': 'sin',
        'other.mid': 'square'}
    for midi_name in os.listdir(midi_directory):
        midi_path = os.path.join(midi_directory, midi_name)
        synthed_stem = process_stem(midi_path,instruments[midi_name])
        sf.write(os.path.join(chiptune_directory,midi_name[:-4]+'.wav'), synthed_stem, samplerate=sr)

    return tracks


# Function that mixes the chiptune stems into the final track
# Input: list of paths to stems
# Returns: the final track
def mixer(tracks,song_name):
    mixed_track = None
    for i, track in enumerate(tracks):
        # load stem
        stem, sr = librosa.load(track)
        # create container for mixing
        if mixed_track is None:
            mixed_track = np.zeros(len(stem))
        # scale bass track
        if 'bass' in track:
            stem = pyrb.pitch_shift(stem, sr, -BASS_SCALE+12)  # 2 octaves
            sf.write(os.path.join('temp_data','chiptune_stems',song_name,'bass.wav'),stem,samplerate=sr)

        if 'other' in track: continue

        # pad if mixed track and curr stem not equal in length
        if len(mixed_track) > len(stem):
            pad_amount = len(mixed_track)-len(stem)
            stem = np.concatenate((stem,np.zeros(pad_amount)))
        elif len(stem) > len(mixed_track):
            pad_amount = len(stem) - len(mixed_track)
            mixed_track = np.concatenate((stem, np.zeros(pad_amount)))
        mixed_track += stem

    return mixed_track




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


def main(audio_file="pop.00000.wav", num_of_stem=4):

    #set up paths and variables needed
    input_song_path = os.path.join("input_songs", audio_file)           # path to source song
    input_song_name = audio_file[:-4]                                   # source song name

    os.makedirs('temp_data', exist_ok=True)                             # make "temp_data" folder to hold files created
    os.makedirs('output', exist_ok=True)                                # make "output" folder

    # load source song
    song_audio, sr = librosa.load(input_song_path)
    bpm = librosa.beat.beat_track(song_audio)[0]

    # BEGIN-----------------------------------------------------
    # PHASE 1: Separate song in to stems
    # stem_separate(song_path=input_song_path, num_stems=num_of_stem)

    # Process stems
    stem_process(song_name=input_song_name,  bpm=bpm, sr=sr)

    # PHASE 2: Convert stem wav files to midi files
    audio2midi(song_name=input_song_name)

    # PHASE 3: Convert midi files to chiptune
    # chiptune_stem_paths = chiptune_synth(song_name=input_song_name, sr=sr)
    chiptune_stem_paths = PySynth_synthisize(song_name=input_song_name, bpm_source=bpm)

    # Mix chiptune stems together
    final_track = mixer(tracks=chiptune_stem_paths, song_name=input_song_name)

    # Output song
    sf.write(os.path.join('output', input_song_name+'_chiptune.wav'), final_track, samplerate=sr)

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
