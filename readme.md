
# Quick start
## install dependencies using conda
conda install -c conda-forge ffmpeg libsndfile
## install spleeter with pip
pip install spleeter
## download an example audio file (if you don't have wget, use another tool for downloading)
wget https://github.com/deezer/spleeter/raw/master/audio_example.mp3
## separate the example audio into two components
spleeter separate -p spleeter:2stems -o output audio_example.mp3
## audio-to-midi tool install
preqres: numpy, librosa, midiutil



To install: pip3 install numpy librosa midiutil

Close the repository into Chiptune-Maker project folder
git clone https://github.com/tiagoft/audio_to_midi.git

audio_to_midi