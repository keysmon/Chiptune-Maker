import os
import sys
import librosa
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from PySynth.pysynth_c import make_wav

# Function to use Spleeter's command line program.
# Not used in current build
def cmdLineSeperator(file):
    os.system("spleeter separate -p spleeter:2stems -o output audio_example.mp3")


# Function to separate instrument stems in a file using Spleeter
# Input: audio file, number of stems {2,4,5}
# Output: None in program, creates desired number of stem wav files in specified path
def stem_separate(file, num_stems=5):
    separator = Separator('spleeter:' + str(num_stems) + 'stems')
    separator.separate_to_file(file, "spleeter_output")

    return


# Function that converts audio files to midi files.
# Input: stem wav file from spleeter
# Output: midi files
def audio2midi(wav_stem, audio2midi_path, file_name, stem_name):
    # create midi file - audio_to_midi needs the file to exist already..
    os.makedirs(os.path.join('midi', file_name), exist_ok=True)
    midi_stem = os.path.join("midi", file_name, stem_name + ".mid")
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


def main(audio_file="audio_example.mp3", num_of_stem=5):
    input_song_path = os.path.join("input_songs", audio_file)
    input_song_name = audio_file[:-4]
    bpm = librosa.beat.tempo(input_song_path)

    # Separate song in to stems
    stem_separate(input_song_path, num_stems=num_of_stem)

    # Convert stem wav files to midi files
    os.makedirs(os.path.join('midi'), exist_ok=True)  # make "midi" folder, if one does not exist
    output_stem_path = os.path.join('spleeter_output', input_song_name)  # path to .wav stem files
    audio2midi_path = os.path.join("audio_to_midi", "audio2midi.py")  # python tool command
    for stem_name in os.listdir(output_stem_path):
        wav_stem = os.path.join(output_stem_path, stem_name)

        audio2midi(wav_stem, audio2midi_path, input_song_name, stem_name[:-4])

    # Convert midi files to chiptune
    midi_path = os.path.join('midi', input_song_name)
    for midi_file in os.listdir(midi_path):
        make_wav(song=os.path.join(midi_path, midi_file), bpm=bpm, fn=midi_file[:-4]+'.wav')






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
