# SRT subtitle timing adjustment

The utility can be useful for people learning a foreign language by watching movies with subtitles.
  If you have a movie in the language you need, then subtitles can be downloaded from the Internet, but
most often these subtitles will be out of sync with the audio.
This utility will help solve this problem. The opensource library is used for recognition
  [Vosk](https://alphacephei.com/vosk/) supporting more than 20 languages and working offline

## Run

To run the utility on your local computer, you must already have Python 3 installed.
as well as the utility [FFmpeg](https://ffmpeg.org/)
You can check if FFmpeg is installed correctly by running the following command in the console: ```ffmpeg -version```


- Download the code.
- Install the dependencies with `pip install -r requirements.txt`.
- Run the utility without parameters with the command: ```python3 srtcorrect.py```
   This will create a ```.conf``` file next to srtcorrect.py with a list of languages and models for their recognition, as well as
   the model for recognition will be downloaded (by default - for the English language),
- If you are interested in another language - uncomment the appropriate line in the ```.conf``` file
- Start subtitle correction with the command:
`python3 srtcorrect.py [-s number_audio_stream] path_to_movie`
   where -s is an optional parameter with the number of the audio track with the desired language (if there is more than one),
    track numbering starts from 0. If the parameter is not specified, the program will try to find
    suitable track on your own

###The necessary conditions:
- For the program to work correctly, it is necessary that the file with subtitles (in SRT format) lies next to the video file and the file names match
- The full path to the video file must not contain two or more consecutive spaces

Possible problems:  
-sometimes an error occurs on long videos  
-the utility does not digest videos with long intervals between conversations  
-tested in English (and a little in Spanish), the rest, in theory, should also work
