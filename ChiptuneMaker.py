import os
import sys
import librosa
from spleeter.separator import Separator
import IPython.display as ipd
import numpy as np
import soundfile as sf
import shutil
import pyrubberband as pyrb
import crepe
from scipy.interpolate import interp1d


BASS_SCALE = 3*12 # octaves


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


# Function to remove excess sound from stem. Gets fundamental frequency
# Input: stem path
# Return: None (original stems replaced)
def stem_process(song_name):
    hop_size = 1024

    stem_directory = os.path.join('temp_data', 'spleeter_output', song_name)  # path to .wav stem files

    for wav_stem in os.listdir(stem_directory):
        if wav_stem == 'drums.wav': continue  #dont sonify drums
        stem_path = os.path.join(stem_directory,wav_stem)
        # load stem
        stem_audio, sr = librosa.load(stem_path)

        # shift the bass up by BASS_SHIFT
        if wav_stem == 'bass.wav':
            stem_audio = pyrb.pitch_shift(stem_audio, sr, BASS_SCALE)

        # predict frequency
        # time, stem_frequency, confidence, activation = crepe.predict(stem, sr, viterbi=True)
        stem_frequency = np.nan_to_num(librosa.pyin(stem_audio, fmin=20, fmax=3000, sr=sr)[0])

        # sonify into audio
        sonified_stem = sonify(stem_frequency, sr, hop_size)

        # covert to original speed
        sonified_stem = librosa.effects.time_stretch(sonified_stem, 2)

        # output for testing
        # sf.write('sonified_'+wav_stem[:-4]+'.wav', sonified_stem, samplerate=sr)

        # overwrite original stem
        sf.write(stem_path, sonified_stem, samplerate=sr)

    return


# Function that converts audio files to midi files.
# Input: stem wav file from spleeter
# Output: midi files
def audio2midi(song_name):

    stem_directory = os.path.join('temp_data', 'spleeter_output', song_name)  # path to .wav stem files
    os.makedirs(os.path.join('temp_data', 'midi'), exist_ok=True)             # create folder for midi files
    os.makedirs(os.path.join('temp_data', 'midi', song_name), exist_ok=True)  # create fold for source song
    audio2midi_path = os.path.join("audio_to_midi", "audio2midi.py")    # path to audio2midi.py tool

    for stem_path in os.listdir(stem_directory):
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
def chiptune_synth(song_name):
    midi_directory = os.path.join('temp_data', 'midi', song_name)                            # midi file directory
    pysynth_path = os.path.join('PySynth','readmidi.py')                                # pysynth path
    os.makedirs(os.path.join('temp_data', 'chiptune_stems', song_name), exist_ok=True)  # create folder for chiptune stems
    tracks = []
    instrument = {      # instruments for each stem. '' is default --syn_a
        'drums.mid': '--syn_p',
        'vocals.mid': '',
        'piano.mid': '--syn_b',
        'bass.mid': '--syn_e',
        'other.mid': ''}
    for midi_file in os.listdir(midi_directory):
        os.system("python " + pysynth_path + " {} {} {} {}".format(os.path.join(midi_directory, midi_file), 1,
                                                                   os.path.join('temp_data', 'chiptune_stems',
                                                                                song_name,
                                                                                midi_file[:-4] + '.wav'),
                                                                   instrument[midi_file]))
        tracks.append(os.path.join('temp_data', 'chiptune_stems', song_name, midi_file[:-4] + '.wav'))
    return tracks


# Function that mixes the chiptune stems into the final track
# Input: list of paths to stems
# Returns: the final track
def mixer(tracks):
    mixed_track = None
    for i, track in enumerate(tracks):
        # load stem
        stem, sr = librosa.load(track)
        # create container for mixing
        if mixed_track is None:
            mixed_track = np.zeros(len(stem))
        # scale bass track
        if 'bass' in track:
            stem = pyrb.pitch_shift(stem, sr, -BASS_SCALE)  # 3 octaves

        if 'other' in track or 'drums' in track: continue

        # pad if mixed track and curr stem not equal in length
        if len(mixed_track) > len(stem):
            pad_amount = len(mixed_track)-len(stem)
            stem = np.concatenate((stem,np.zeros(pad_amount)))
        elif len(stem) > len(mixed_track):
            pad_amount = len(stem) - len(mixed_track)
            mixed_track = np.concatenate((stem, np.zeros(pad_amount)))
        mixed_track += stem

    return mixed_track


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


def main(audio_file="pop.00000.wav", num_of_stem=4):
    #set up paths and variables needed
    input_song_path = os.path.join("input_songs", audio_file)           # path to source song
    input_song_name = audio_file[:-4]                                   # source song name

    os.makedirs('temp_data', exist_ok=True)                             # make "temp_data" folder to hold files created
    os.makedirs('output', exist_ok=True)                                # make "output" folder

    # load source song
    song_audio, sr = librosa.load(input_song_path)
    bpm = librosa.beat.tempo(song_audio)

    # BEGIN-----------------------------------------------------
    # PHASE 1: Separate song in to stems
    stem_separate(song_path=input_song_path, num_stems=num_of_stem)

    # Process stems
    stem_process(song_name=input_song_name)

    # PHASE 2: Convert stem wav files to midi files
    audio2midi(song_name=input_song_name)

    # PHASE 3: Convert midi files to chiptune
    chiptune_stem_paths = chiptune_synth(song_name=input_song_name)

    # Mix chiptune stems together
    final_track = mixer(tracks=chiptune_stem_paths)

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
