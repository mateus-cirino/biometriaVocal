from librosa import display
from librosa import core
from scipy import fft
from matplotlib import pyplot

import numpy as np

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
    samples, samplingRate = core.load(
                                file_path,
                                sr=None,
                                mono=True,
                                offset=0.0,
                                duration=None
                                )
    # samples contém um array de amostras
    # samplingRate contém a quantidade de amostras capturadas por segundo
    # logo, podemos concluir que a duração de um áudio é igual a quantidadeDeAmostras/quantidadeDeAmostrasPorSegundo
    arrOfAudios.append([samples,samplingRate])
    numberOfAudio += 1

samplesOfAudio1 = arrOfAudios[0][0]
samplesRateOfAudio1 = arrOfAudios[0][1]

#Exibição do gráfico que representa o sinal do áudio no dominio da amplitude
pyplot.figure()
display.waveplot(y = samplesOfAudio1, sr = samplesRateOfAudio1)
pyplot.xlabel("Time (seconds) --> ")
pyplot.ylabel("Amplitude")
pyplot.show()

#Exibição do gráfico que representa o sinal do áudio no dominio da frequência
fftOut = fft(samplesOfAudio1)
pyplot.figure()
pyplot.plot(samplesOfAudio1, np.abs(fftOut))
pyplot.xlabel("Frequence")
pyplot.ylabel("Magnitude")
pyplot.show()


