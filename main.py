import librosa

file_extension = ".ogg"

arr = []

i = 1

while i < 34:
    file_path = str(i) + file_extension
    samples, sampling_rate = librosa.load(
        file_path,
        sr=None,
        mono=True,
        offset=0.0,
        duration=None
    )
    arr.append([samples,sampling_rate])
    i += 1
print(len(arr))
