import os
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

def inAppSeperator(file, num_stems=5):
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(file, "spleeter_stems")

def cmdLineSeperator(file):
    os.system("spleeter separate -p spleeter:2stems -o output audio_example.mp3")


def main():
    audio_file = 'audio_example.mp3'
    input_path = os.path.join("input_songs", audio_file)

    # Separate song in to stems
    inAppSeperator(input_path, num_stems=5)

    # modify stem wav files to midi files
    os.makedirs(os.path.join('midi'), exist_ok=True)    # make "midi" folder, if one does not exist
    output_stem_path = os.path.join('spleeter_stems', audio_file[:-4])   # path to .wav stem files
    audio2midi_path = os.path.join("audio_to_midi", "audio2midi.py") # python tool command
    for stem_name in os.listdir(output_stem_path):
        wav_stem = os.path.join(output_stem_path, stem_name)

        # create output midi file
        os.makedirs(os.path.join('midi', audio_file[:-4]), exist_ok=True)
        midi_stem = os.path.join("midi", audio_file[:-4],stem_name[:-4]+".mid")
        output_midi = open(midi_stem, 'w')

        # call audio_to_midi
        os.system("python " + audio2midi_path + " {} {}".format(wav_stem, midi_stem))
        #os.system("python3 " + audio2midi_path + " {} {}".format(wav_stem, midi_stem))

        output_midi.close()

    return


if __name__ == "__main__":
    main()
