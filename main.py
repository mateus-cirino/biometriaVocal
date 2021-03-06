from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from librosa import core

from scipy import fft

import csv

import math

import os

# SESSAO TREINAMENTO
VERSAO = "48000-100-100-10-5-5"
NETWORK = buildNetwork(48000, 100, 100, 10, 5, 5)

# SESSAO TESTES
# VERSAO = "48000-100-100-10-5-TESTE-AUDIO-JOAS"
# NETWORK = NetworkReader.readFrom('resultado/48000-100-100-10-5/network.xml');

# OUTROS
CAMINHO_RESULTADO = "resultado/" + VERSAO + "/"
EXTENSAO_ARQUIVO_AUDIO = ".ogg"
NOME_DOS_CAMPOS_VALIDACAO = ["Nome do arquivo",
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
DATASET = SupervisedDataSet(48000, 5)


# Função que recebe o caminho da pasta e número de áudios que a pasta contem e retorna
# um array contendo todas as amostras dos áudios na pasta
def carregarAudios(pasta, numDeAudiosNaPasta):
    numeroDeAudios = 1
    arrayDeAmostras = []

    while numeroDeAudios <= numDeAudiosNaPasta:
        caminhoDoArquivo = pasta + str(numeroDeAudios) + EXTENSAO_ARQUIVO_AUDIO
        amostras, numeroDeAmostrasPorSegundo = core.load(caminhoDoArquivo)
        arrayDeAmostras.extend(amostras)
        numeroDeAudios += 1

    return arrayDeAmostras


# Função que recebe um array de amostras e retorna um array de amostras após o processamento de fft
def aplicarFFTnasAmostras(arrayDeAmostras):
    arrayDeAmostrasAposProcessamentoFFT = fft(arrayDeAmostras)

    return arrayDeAmostrasAposProcessamentoFFT


# Função que retorna um array com cada posição do mesmo contendo 1 segundo = 48000
def separacaoEmSegundosDoAudio(arrayDeAmostras):
    arrayDeSegundoDosAudios = []
    totalDeSegundos = int(len(arrayDeAmostras) / 48000)
    i = 1
    while totalDeSegundos >= i:
        arrayDeSegundoDosAudios.append(arrayDeAmostras[48000 * (i - 1):48000 * i])
        i += 1

    return arrayDeSegundoDosAudios


# Função para treinamento da Rede Neural Artificial
def treinamentoRedeNeural(rede, dados, nomeDoArquivoDeResultados):
    trainer = BackpropTrainer(rede, dados)
    iteration = 0
    error = 1

    with open(nomeDoArquivoDeResultados, mode='w+', newline='') as csv_file:
        nomeDosCampos = ["Iteração",
                         "Erro"
                         ]
        writer = csv.DictWriter(csv_file, fieldnames=nomeDosCampos)
        writer.writeheader()
        while error > 0.0001:
            error = trainer.train()
            writer.writerow(
                {"Iteração": iteration,
                 "Erro": error})
            print("i = ", iteration, " erro = ", error)
            iteration += 1


# Função para a validação da Rede Neural Artificial
def validacaoRedeNeural(rede, dados, nomeDoArquivo, valoresEsperados, nomeDoArquivoDeResultados):
    with open(nomeDoArquivoDeResultados, mode='w+', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=NOME_DOS_CAMPOS_VALIDACAO)
        writer.writeheader()
        i = 0
        for segundo in dados:
            i += 1
            saidaDaRedeNeural = rede.activate(segundo)
            writer.writerow(
                {
                    NOME_DOS_CAMPOS_VALIDACAO[0]: nomeDoArquivo + EXTENSAO_ARQUIVO_AUDIO,
                    NOME_DOS_CAMPOS_VALIDACAO[1]: str(i),
                    NOME_DOS_CAMPOS_VALIDACAO[2]: valoresEsperados[0],
                    NOME_DOS_CAMPOS_VALIDACAO[3]: round(saidaDaRedeNeural[0], 2),
                    NOME_DOS_CAMPOS_VALIDACAO[4]: round((saidaDaRedeNeural[0] - valoresEsperados[0]), 2),
                    NOME_DOS_CAMPOS_VALIDACAO[5]: valoresEsperados[1],
                    NOME_DOS_CAMPOS_VALIDACAO[6]: round(saidaDaRedeNeural[1], 2),
                    NOME_DOS_CAMPOS_VALIDACAO[7]: round((saidaDaRedeNeural[1] - valoresEsperados[1]), 2),
                    NOME_DOS_CAMPOS_VALIDACAO[8]: valoresEsperados[2],
                    NOME_DOS_CAMPOS_VALIDACAO[9]: round(saidaDaRedeNeural[2], 2),
                    NOME_DOS_CAMPOS_VALIDACAO[10]: round((saidaDaRedeNeural[2] - valoresEsperados[2]), 2),
                    NOME_DOS_CAMPOS_VALIDACAO[11]: valoresEsperados[3],
                    NOME_DOS_CAMPOS_VALIDACAO[12]: round(saidaDaRedeNeural[3], 2),
                    NOME_DOS_CAMPOS_VALIDACAO[13]: round((saidaDaRedeNeural[3] - valoresEsperados[3]), 2),
                    NOME_DOS_CAMPOS_VALIDACAO[14]: valoresEsperados[4],
                    NOME_DOS_CAMPOS_VALIDACAO[15]: round(saidaDaRedeNeural[4], 2),
                    NOME_DOS_CAMPOS_VALIDACAO[16]: round((saidaDaRedeNeural[4] - valoresEsperados[4]), 2),
                    NOME_DOS_CAMPOS_VALIDACAO[17]: round(math.sqrt(((
                                                                            math.pow((saidaDaRedeNeural[0] -
                                                                                      valoresEsperados[0]), 2) +
                                                                            math.pow((saidaDaRedeNeural[1] -
                                                                                      valoresEsperados[1]), 2) +
                                                                            math.pow((saidaDaRedeNeural[2] -
                                                                                      valoresEsperados[2]), 2) +
                                                                            math.pow((saidaDaRedeNeural[3] -
                                                                                      valoresEsperados[3]), 2) +
                                                                            math.pow((saidaDaRedeNeural[4] -
                                                                                      valoresEsperados[4]), 2)) / 5)),
                                                         2)
                }
            )

# SESSAO 1 => TREINAMENTO

# CRIANDO OS DIRETORIOS
os.mkdir("resultado/" + VERSAO)

# TREINAMENTO
audios = carregarAudios("caminho da pasta com os audios de treinamento do falante 1/", 1)

audiosAposFFT = aplicarFFTnasAmostras(audios)

arrayComOsSegundosDosAudios = separacaoEmSegundosDoAudio(audiosAposFFT)

for segundo in arrayComOsSegundosDosAudios:
    DATASET.addSample(segundo, (10, 10, 10, 10, 10)) # os valores (10, 10, 10, 10, 10) foram escolhidos para representar o falante 1

audios = carregarAudios("caminho da pasta com os audios de treinamento do falante 2/", 1)

audiosAposFFT = aplicarFFTnasAmostras(audios)

arrayComOsSegundosDosAudios = separacaoEmSegundosDoAudio(audiosAposFFT)

for segundo in arrayComOsSegundosDosAudios:
    DATASET.addSample(segundo, (20, 20, 20, 20, 20)) # os valores (20, 20, 20, 20, 20) foram escolhidos para representar o falante 2

treinamentoRedeNeural(NETWORK, DATASET, (CAMINHO_RESULTADO + "TREINAMENTO.csv"))

# VALIDACAO
audio = carregarAudios("caminho da pasta com os audios de validacao do falante 1/", 1)

audioAposFFT = aplicarFFTnasAmostras(audio)

arrayComOsSegundosDoAudio = separacaoEmSegundosDoAudio(audioAposFFT)

validacaoRedeNeural(NETWORK, arrayComOsSegundosDoAudio, "1.ogg", [10, 10, 10, 10, 10],
                    (CAMINHO_RESULTADO + "VALIDACAO.csv"))

NetworkWriter.writeToFile(NETWORK, (CAMINHO_RESULTADO + "network.xml"))


# SESSAO 2 => TESTE

# # CRIANDO OS DIRETORIOS
# os.mkdir("resultado/" + VERSAO)
#
# # VALIDACAO
# audio = carregarAudios("caminho da pasta com os audios de validacao do falante x/", 1)
#
# audioAposFFT = aplicarFFTnasAmostras(audio)
#
# arrayComOsSegundosDoAudio = separacaoEmSegundosDoAudio(audioAposFFT)
#
# validacaoRedeNeural(NETWORK, arrayComOsSegundosDoAudio, "1.ogg", [10, 10, 10, 10, 10],
#                     (CAMINHO_RESULTADO + "VALIDACAO.csv"))
