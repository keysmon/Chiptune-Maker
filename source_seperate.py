import os




def main():
    audio_file = "audio_example.mp3"
    
    
    os.system("spleeter separate -p spleeter:2stems -o output audio_example.mp3" )
    return
    
    
    
if __name__ == "__main__":
    main()
