from mido import MidiFile
from PySynth.pysynth_c import make_wav
import sys

def main():
    print("Hello")

    #sys.path.insert(0, './midi_files')


    mid = MidiFile('./midi_files/Am_I_Blue_AB.mid', clip=True)
    
    make_wav(mid, fn = "test.wav")
    
    print(mid)


if __name__ == "__main__":
    main()