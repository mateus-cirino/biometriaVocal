from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from librosa import core

from scipy import fft

import csv

import math

import os
# ATENÇÃO: ESSA VERSÃO DO CÓDIGO NÃO GERA ARQUIVOS CSV DE RESULTADOS (TREINAMENTO E VALIDAÇÃO)
# CASO VOCÊ QUEIRA A GERAÇÃO DOS CSV UTILIZE O ARQUIVO MAIN.PY, ESSE ARQUIVO FOI CRIADO ÚNICO
# E EXCLUSIVAMENTE PARA SUPORTE DE CÓDIGO PARA O ARTIGO DE TCC

# SESSAO TREINAMENTO
VERSAO = "48000-100-100-10-5-5"
NETWORK = buildNetwork(48000, 100, 100, 10, 5, 5)

# SESSAO TESTES
# VERSAO = "48000-100-100-10-5-TESTE-AUDIO-JOAS"
# NETWORK = NetworkReader.readFrom('resultado/48000-100-100-10-5/network.xml');

# OUTROS
CAMINHO_RESULTADO = "resultado/" + VERSAO + "/"
EXTENSAO_ARQUIVO_AUDIO = ".ogg"
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
def treinamentoRedeNeural(rede, dados):
    trainer = BackpropTrainer(rede, dados)
    error = 1

    while error > 0.0001:
        error = trainer.train()


# Função para a validação da Rede Neural Artificial
def validacaoRedeNeural(rede, dados, nomeDoArquivo, valoresEsperados, nomeDoArquivoDeResultados):
    for segundo in dados:
        saidaDaRedeNeural = rede.activate(segundo)

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
