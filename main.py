import librosa

file_extension = ".ogg"

arrOfAudios = []

numberOfAudio = 1

    # numberOfAudio < 34, pois eu estou lendo 33 áudios que se encontram na minha pasta local
while numberOfAudio < 34:
    file_path = str(numberOfAudio) + file_extension
    #VARIAVEIS
    # sr (caso eu queira mudar a taxa de amostragem)
    # mono (se o áudio é mono ou stereo)
    # offset (caso eu queira começar de uma determinada parte do áudio)
    # duration (caso eu queira pegar somente uma parte do áudio)
    samples, sampling_rate = librosa.load(
                                file_path,
                                sr=None,
                                mono=True,
                                offset=0.0,
                                duration=None
                                )
    # samples contém um array de amostras
    # sampling_rate contém a quantidade de amostras capturadas por segundo
    # logo, podemos concluir que a duração de um áudio é igual a quantidadeDeAmostras/quantidadeDeAmostrasPorSegundo
    arrOfAudios.append([samples,sampling_rate])
    numberOfAudio += 1
print(len(arrOfAudios))
