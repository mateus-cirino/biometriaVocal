from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from librosa import core

from scipy import fft

import csv

extensaoDoArquivo = ".ogg"

network = buildNetwork(48000, 100, 100, 5)
network = NetworkReader.readFrom('network.xml');

dataSet = SupervisedDataSet(48000, 5)

numeroDeAudios = 16
arrayDeFFTDosAudios = []

while numeroDeAudios < 23:
    caminhoDoArquivo = "audiosMateusTreinamento/" + str(numeroDeAudios) + extensaoDoArquivo
    amostras, numeroDeAmostrasPorSegundo = core.load(caminhoDoArquivo, sr=None, mono=True, offset=0.0, duration=None)
    arrayDeFFTDosAudios.extend(fft(amostras))
    numeroDeAudios += 1

while len(arrayDeFFTDosAudios) > 48000:
    amostrasPorSegundo = arrayDeFFTDosAudios[:48000]
    del arrayDeFFTDosAudios[:48000]
    dataSet.addSample(amostrasPorSegundo, (10, 10, 10, 10, 10))

trainer = BackpropTrainer(network, dataSet)
iteration = 0
error = 1
while error > 0.001:
    error = trainer.train()
    print(iteration, error)
    iteration += 1

NetworkWriter.writeToFile(network, 'network.xml')

numeroDeAudios = 1
arrayDeFFTDosAudios = [];

while numeroDeAudios < 4:
    caminhoDoArquivo = "audiosMateusValidacao/" + str(numeroDeAudios) + extensaoDoArquivo
    amostras, numeroDeAmostrasPorSegundo = core.load(caminhoDoArquivo, sr=None, mono=True, offset=0.0, duration=None)
    arrayDeFFTDosAudios.extend(fft(amostras))
    numeroDeAudios += 1

with open('treinamentoRedeNeural.csv', mode='w', newline='') as csv_file:
    nomeDosCampos = ["Nome do arquivo", "Valor esperado", "Valor obtido", "Erro"]
    writer = csv.DictWriter(csv_file, fieldnames=nomeDosCampos)

    writer.writeheader()
    numeroDeAudios = 1
    while len(arrayDeFFTDosAudios) > 48000:
        amostrasPorSegundo = arrayDeFFTDosAudios[:48000]
        del arrayDeFFTDosAudios[:48000]
        saidaDaRedeNeural = network.activate(amostrasPorSegundo)
        writer.writerow({"Nome do arquivo": (str(numeroDeAudios) + ".ogg"), "Valor esperado": 10, "Valor obtido": saidaDaRedeNeural[0], "Erro": (saidaDaRedeNeural[0] - 10)})
        writer.writerow({"Nome do arquivo": (str(numeroDeAudios) + ".ogg"), "Valor esperado": 10, "Valor obtido": saidaDaRedeNeural[1], "Erro": (saidaDaRedeNeural[1] - 10)})
        writer.writerow({"Nome do arquivo": (str(numeroDeAudios) + ".ogg"), "Valor esperado": 10, "Valor obtido": saidaDaRedeNeural[2], "Erro": (saidaDaRedeNeural[2] - 10)})
        writer.writerow({"Nome do arquivo": (str(numeroDeAudios) + ".ogg"), "Valor esperado": 10, "Valor obtido": saidaDaRedeNeural[3], "Erro": (saidaDaRedeNeural[3] - 10)})
        writer.writerow({"Nome do arquivo": (str(numeroDeAudios) + ".ogg"), "Valor esperado": 10, "Valor obtido": saidaDaRedeNeural[4], "Erro": (saidaDaRedeNeural[4] - 10)})
        writer.writerow({"Nome do arquivo": "", "Valor esperado": "", "Valor obtido": "", "Erro": ""})
        numeroDeAudios += 1
