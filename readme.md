
# Quick start
## install dependencies using conda
conda install -c conda-forge ffmpeg libsndfile
## install spleeter with pip
pip install spleeter
## download an example audio file (if you don't have wget, use another tool for downloading)
wget https://github.com/deezer/spleeter/raw/master/audio_example.mp3
## separate the example audio into two components
spleeter separate -p spleeter:2stems -o output audio_example.mp3
