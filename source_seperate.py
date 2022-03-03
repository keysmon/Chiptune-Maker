import os
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

def inAppSeperator(file):
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(file,"output")

def cmdLineSeperator(file):
    os.system("spleeter separate -p spleeter:2stems -o output audio_example.mp3")


def main():
    audio_file = "input_songs\\vocals_guitar.mp3"

    inAppSeperator(audio_file)
    #cmdLineSeperator(audio_file)

    return


if __name__ == "__main__":
    main()
