import os
import sys
import librosa
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import IPython.display as ipd
import numpy as np
import soundfile as sf
import shutil

# Function to use Spleeter's command line program.
# Not used in current build
def cmdLineSeperator(file):

    os.system("spleeter separate -p spleeter:2stems -o output audio_example.mp3")


# Function to separate instrument stems in a file using Spleeter
# Input: audio file, number of stems {2,4,5}
# Output: None in program, creates desired number of stem wav files in specified path
def stem_separate(file, num_stems=5):
    try:
        shutil.move(os.path.join('temp_data', 'pretrained_models'), os.getcwd())             #move spleeter helper folder to main folder
    except:
        pass

    separator = Separator('spleeter:' + str(num_stems) + 'stems')
    separator.separate_to_file(file, os.path.join("temp_data", "spleeter_output"))

    shutil.move('pretrained_models', 'temp_data')            #move it back

    return


# TODO: this is not capturing bass properly
# Function that converts audio files to midi files.
# Input: stem wav file from spleeter
# Output: midi files
def audio2midi(wav_stem, audio2midi_path, file_name, stem_name):
    # create midi file - audio_to_midi needs the file to exist already..
    os.makedirs(os.path.join('temp_data', 'midi', file_name), exist_ok=True)
    midi_stem = os.path.join('temp_data', "midi", file_name, stem_name + ".mid")
    output_midi = open(midi_stem, 'w')

    try:        #using the command line tool for audio_to_midi.
        os.system("python " + audio2midi_path + " {} {}".format(wav_stem, midi_stem))
    except:
        try:
            os.system("python3 " + audio2midi_path + " {} {}".format(wav_stem, midi_stem))
        except:
            print("Error with audio2midi package.")
            sys.exit()

    output_midi.close()
    return

def mixer(tracks):
    track = librosa.load(tracks[0])[0]
    for t in range(1,len(tracks)):
        stem = librosa.load(tracks[t])[0]
        # pad instrument tracks if too short
        if len(track) > len(stem):
            pad_amount = len(track)-len(stem)
            stem = np.concatenate((stem,np.zeros(pad_amount)))
        elif len(stem) > len(track):
            pad_amount = len(stem) - len(track)
            track = np.concatenate((stem, np.zeros(pad_amount)))
        track += stem

    return track


def main(audio_file="pop.00000.wav", num_of_stem=4):
    input_song_path = os.path.join("input_songs", audio_file)
    input_song_name = audio_file[:-4]
    song, sr = librosa.load(input_song_path)
    bpm = librosa.beat.tempo(song)

    # create data file for storage of temp files
    os.makedirs('temp_data', exist_ok=True)

    # Separate song in to stems
    stem_separate(input_song_path, num_stems=num_of_stem)

    sys.exit()

    # Convert stem wav files to midi files
    os.makedirs(os.path.join('temp_data', 'midi'), exist_ok=True)  # make "midi" folder, if one does not exist
    output_stem_path = os.path.join('temp_data', 'spleeter_output', input_song_name)  # path to .wav stem files
    audio2midi_path = os.path.join("audio_to_midi", "audio2midi.py")  # python tool command
    for stem_name in os.listdir(output_stem_path):
        wav_stem = os.path.join(output_stem_path, stem_name)

        audio2midi(wav_stem, audio2midi_path, input_song_name, stem_name[:-4])

    # Convert midi files to chiptune
    midi_path = os.path.join('temp_data', 'midi', input_song_name)
    pysynth_path = 'PySynth/readmidi.py'
    os.makedirs(os.path.join('temp_data', 'chiptune_stems', input_song_name), exist_ok=True)
    tracks = []
    instrument = {
        'drums.mid': '--syn_p',
        'vocals.mid': '',
        'piano.mid': '--syn_b',
        'bass.mid': '--syn_e',
        'other.mid': ''}
    for midi_file in os.listdir(midi_path):
        os.system("python " + pysynth_path + " {} {} {} {}".format(os.path.join(midi_path,midi_file), 1, os.path.join('temp_data', 'chiptune_stems', input_song_name, midi_file[:-4]+'.wav'), instrument[midi_file]))
        tracks.append(os.path.join('temp_data', 'chiptune_stems', input_song_name,midi_file[:-4]+'.wav'))


    final_track = mixer(tracks)
    sf.write(input_song_name+'_chiptune.wav', final_track, samplerate=sr)

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
