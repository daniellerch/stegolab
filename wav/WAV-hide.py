#!/usr/bin/python3

import wave

cover = wave.open("file.wav", mode='rb')
frames = bytearray(cover.readframes( cover.getnframes() ))

print("frames:", len(frames))


# Modify the frame 7 
frames[7] += 1

with wave.open('file_stego.wav', 'wb') as stego:
    stego.setparams(cover.getparams())
    stego.writeframes(bytes(frames))

cover.close()

