from pybrain.tools.customxml import NetworkWriter
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from librosa import core

from scipy import fft

import matplotlib.pyplot as plt

extensaoDoArquivo = ".ogg"

arrayDeFFTDosAudios = []

numeroDeAudios = 1

while numeroDeAudios < 16:
    caminhoDoArquivo = str(numeroDeAudios) + extensaoDoArquivo
    amostras, numeroDeAmostrasPorSegundo = core.load(caminhoDoArquivo, sr=None, mono=True, offset=0.0, duration=None)
    arrayDeFFTDosAudios.extend(fft(amostras))
    numeroDeAudios += 1

network = buildNetwork(48000, 100, 100, 5)
dataSet = SupervisedDataSet(48000, 5)

while len(arrayDeFFTDosAudios) > 48000:
    amostrasPorSegundo = arrayDeFFTDosAudios[:48000]
    del arrayDeFFTDosAudios[:48000]
    dataSet.addSample(amostrasPorSegundo, (10, 10, 10, 10, 10))

trainer = BackpropTrainer(network, dataSet)
error = 1
iteration = 0
outputs = []
while error > 0.001:
    error = trainer.train()
    outputs.append(error)
    iteration += 1
    print(iteration, error)

NetworkWriter.writeToFile(network, 'network.xml')

plt.ioff()
plt.plot(outputs)
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()


