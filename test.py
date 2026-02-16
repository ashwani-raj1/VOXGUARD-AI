from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

audio = AudioSegment.from_file("test.mp3")
audio.export("out.wav", format="wav", parameters=["-acodec", "pcm_s16le"])

print("Converted")
