import os
import sys
import vamp
import librosa


def main(audio_file = "audio_example.mp3",num_of_stem = 2):
    
    
    command = "spleeter separate -p spleeter:"+str(num_of_stem)+"stems -o output " + audio_file
    os.system(command)
    audio,sr = librosa.load("output/audio_example/vocals.wav")
    data = vamp.collect(audio,sr,"mtg-melodia:melodia")
    print(data)
    return
    
    
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1],sys.argv[2])
    else:
        println("Error: Invalid Args!")
     
