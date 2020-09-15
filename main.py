from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from librosa import core

from scipy import fft

import csv

import math

extensaoDoArquivo = ".ogg"

network = buildNetwork(48000, 100, 100, 5)
# network = NetworkReader.readFrom('network.xml');

dataSet = SupervisedDataSet(48000, 5)

numeroDeAudios = 1
arrayDeFFTDosAudios = []

while numeroDeAudios < 17:
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

numeroDeAudios = 1
arrayComADuracaoDosAudios = []
arrayDeFFTDosAudios = []

while numeroDeAudios < 17:
    caminhoDoArquivo = "audiosMateusValidacao/" + str(numeroDeAudios) + extensaoDoArquivo
    amostras, numeroDeAmostrasPorSegundo = core.load(caminhoDoArquivo, sr=None, mono=True, offset=0.0, duration=None)
    duracaoEmSegundos = int(len(amostras)/numeroDeAmostrasPorSegundo)
    arrayComADuracaoDosAudios.append(duracaoEmSegundos)
    arrayDeFFTDosAudios.extend(fft(amostras))
    numeroDeAudios += 1

with open('treinamentoRedeNeural.csv', mode='w+', newline='') as csv_file:
    nomeDosCampos = ["Nome do arquivo",
                     "Segundo",
                     "Valor esperado 1",
                     "Valor obtido 1",
                     "Erro 1",
                     "Valor esperado 2",
                     "Valor obtido 2",
                     "Erro 2",
                     "Valor esperado 3",
                     "Valor obtido 3",
                     "Erro 3",
                     "Valor esperado 4",
                     "Valor obtido 4",
                     "Erro 4",
                     "Valor esperado 5",
                     "Valor obtido 5",
                     "Erro 5",
                     "Erro quadrático médio"
                     ]
    writer = csv.DictWriter(csv_file, fieldnames=nomeDosCampos)
    writer.writeheader()
    i = 0
    while i < len(arrayComADuracaoDosAudios):
        j = 0
        while j < arrayComADuracaoDosAudios[i]:
            amostrasPorSegundo = arrayDeFFTDosAudios[:48000]
            del arrayDeFFTDosAudios[:48000]
            saidaDaRedeNeural = network.activate(amostrasPorSegundo)
            writer.writerow(
                {
                        "Nome do arquivo": (str(i + 1) + ".ogg"),
                        "Segundo": (str(j + 1)),
                        "Valor esperado 1": 10.00,
                        "Valor obtido 1": round(saidaDaRedeNeural[0], 2),
                        "Erro 1": round((saidaDaRedeNeural[0] - 10), 2),
                        "Valor esperado 2": 10.00,
                        "Valor obtido 2": round(saidaDaRedeNeural[1], 2),
                        "Erro 2": round((saidaDaRedeNeural[1] - 10), 2),
                        "Valor esperado 3": 10.00,
                        "Valor obtido 3": round(saidaDaRedeNeural[2], 2),
                        "Erro 3": round((saidaDaRedeNeural[2] - 10), 2),
                        "Valor esperado 4": 10.00,
                        "Valor obtido 4": round(saidaDaRedeNeural[3], 2),
                        "Erro 4": round((saidaDaRedeNeural[3] - 10), 2),
                        "Valor esperado 5": 10.00,
                        "Valor obtido 5": round(saidaDaRedeNeural[4], 2),
                        "Erro 5": round((saidaDaRedeNeural[4] - 10), 2),
                        "Erro quadrático médio": round(math.sqrt(((math.pow((saidaDaRedeNeural[0] - 10), 2) + math.pow((saidaDaRedeNeural[1] - 10), 2)
                                                             + math.pow((saidaDaRedeNeural[2] - 10), 2) + math.pow((saidaDaRedeNeural[3] - 10), 2)
                                                             + math.pow((saidaDaRedeNeural[4] - 10), 2))/5)), 2)
                }
            )
            j += 1
        i += 1
        writer.writerow(
            {
                "Nome do arquivo": "",
                "Segundo": "",
                "Valor esperado 1": "",
                "Valor obtido 1": "",
                "Erro 1": "",
                "Valor esperado 2": "",
                "Valor obtido 2": "",
                "Erro 2": "",
                "Valor esperado 3": "",
                "Valor obtido 3": "",
                "Erro 3": "",
                "Valor esperado 4": "",
                "Valor obtido 4": "",
                "Erro 4": "",
                "Valor esperado 5": "",
                "Valor obtido 5": "",
                "Erro 5": "",
                "Erro quadrático médio": ""
            }
        )
NetworkWriter.writeToFile(network, 'network.xml')

