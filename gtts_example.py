# Озвучка текста

# dependency: pip install gTTS 
# https://gtts.readthedocs.io/en/latest/

'''
Use cli:
$ gtts-cli 'Привет, Евгений!' --output source/msg.mp3
'''
from gtts import gTTS
from playsound import playsound

tts = gTTS("Привет, Евгений!",lang="ru")
tts.save("source/msg.mp3")
playsound("source/msg.mp3")

# python gtts_example.py