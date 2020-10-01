#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax

import sys
DEBUG = False
if len(sys.argv) > 1 and sys.argv[1] == "debuga":
    DEBUG = True

def print_debug(x):
    if DEBUG:
        print(x)

def sigmoid(x):
    """
    Computa a função sigmóide para a entrada.
     Argumentos:
     x - um escalar ou um numpy array
     Retorna:
     s - sigmóide (x)
    """

    ### Seu código aqui (~1 Linha)
    s = 1 / ( 1 + np.exp(-x) )
    ### Fim do seu código

    return s


def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors,
                                dataset):
    """  Função de gradiente & Naive (ingênuo) Softmax loss (custo) para modelos word2vec

    Implementar a softmax naive loss e gradientes entre o vetor de uma palavra central
    e o vetor de uma palavra externa. Este será o bloco de construção para nossos modelos word2vec.

    Argumentos:
    centerWordVec - numpy ndarray, vetor da palavra central
                    com shape (comprimento do vetor da palavra,)
                    (v_c no enunciado pdf)
    outsideWordIdx - inteiro, o índice da palavra externa
                    (o de u_o no enunciado pdf)
    outsideVectors - matriz de vetores externos com shape (número de palavras no vocabulário, dimensão do embedding)
                    para todas as palavras do vocabulário, cada linha um vetor (U (|V| x n) no folheto em pdf)
    dataset - necessário para amostragem negativa, não utilizado aqui.

    Retorna:
    loss  -  naive softmax loss
    gradCenterVec - o gradiente em relação ao vetor da palavra central
                     com shape (dimensão do embedding,)
                     (dJ / dv_c no enunciado pdf)
    gradOutsideVecs - o gradiente em relação a todos os vetores de palavras externos
                    com shape (num palavras no vocabulário, dimensão do embedding)
                    (dJ / dU)
    """

    ### Seu código aqui (~6-8 Lines)

    ### Use a função softmax fornecida (importada anteriormente neste arquivo)
    ### Esta implementação numericamente estável ajuda a evitar problemas causados
    ### por estouro de inteiro (integer overflow).

    # outsideVectors = np.array([(1,2,3), (4,5,6), (7,8,9), (10,11,12)], dtype = float)
    # v_c = centerWordVec
    # U = outsideVectors
    V_size = outsideVectors.shape[0]

    # vc_UT = centerWordVec.dot(outsideVectors.T)
    y_hat = softmax( centerWordVec.dot( outsideVectors.T ) )
    
    loss = -np.log( y_hat[outsideWordIdx] ) # O loss entre "c" e "o", é -log( y_hat[o])

    y = np.zeros(V_size)
    y[outsideWordIdx] = 1

    y_diff = y_hat - y
    # dJ_dv = gradCenterVec
    gradCenterVec = outsideVectors.T.dot( y_diff ) 

    y_hat[ outsideWordIdx ] = y_hat[ outsideWordIdx ] - 1

    gradOutsideVecs = np.zeros( outsideVectors.shape )
    for o in range(V_size):
        gradOutsideVecs[o] = centerWordVec.dot(  y_diff[o] )


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Amostra K indices distintos de outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec,
                               outsideWordIdx,
                               outsideVectors,
                               dataset,
                               K=10):
    """ Função de custo (loss) de amostragem negativa para modelos word2vec

     Implemente o custo de amostragem negativa e gradientes para um vetor de palavra centerWordVec
     e um vetor de palavra outsideWordIdx.
     K é o número de amostras negativas a serem colhidas.

     Observação: a mesma palavra pode ser amostrada negativamente várias vezes. Por
     exemplo, se uma palavra externa for amostrada duas vezes, você deverá
     contar duas vezes o gradiente em relação a esta palavra. Três vezes se
     foi amostrado três vezes e assim por diante.

     Argumentos / especificações de devolução: iguais a naiveSoftmaxLossAndGradient
     """

    # A amostragem negativa de palavras está pronta para você.
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### Seu código aqui  (~10 Lines)
    uo = outsideVectors[outsideWordIdx]
    
    # Calculando a perda: ver o questão (e): J_amostranegativa
    uo_vc = uo.dot(centerWordVec)
    loss_p1 = - np.log( sigmoid( uo_vc ) )
    loss_p2 = - np.sum( np.log( sigmoid( - outsideVectors[negSampleWordIndices].dot(centerWordVec) ) )  )
    loss = loss_p1 + loss_p2

    # Calculando a respota (e.1) do PDF: derivada de J_amostranegativa
    uk = outsideVectors[negSampleWordIndices]
    uk_vc = uk.dot(centerWordVec)

    dj_vc_p1 = -uo * (1 - sigmoid(uo_vc)) 
    dj_vc_p2 = np.expand_dims(1 - sigmoid(-uk_vc), axis=1) * uk
    dj_vc_p2 = np.sum(dj_vc_p2, axis=0)

    gradCenterVec = dj_vc_p1 + dj_vc_p2

    # Calculando a respota (e.2 e e.3) do PDF: derivada de J em relação a u_o e u_k.
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    # Observação: a mesma palavra pode ser amostrada negativamente várias vezes. Por
    # exemplo, se uma palavra externa for amostrada duas vezes, você deverá
    # contar duas vezes o gradiente em relação a esta palavra. Três vezes se
    # foi amostrado três vezes e assim por diante.
    for k, idx_k in enumerate(negSampleWordIndices):
        gradOutsideVecs[idx_k] = gradOutsideVecs[idx_k] + (1 - sigmoid(-uk_vc)[k]) * centerWordVec
    
    # Calculando a respota (e.2) do PDF: derivada de J em relação a u_o. Lembrando que "o not in k", 
    # logo no loop anterior o "o" não foi calculado.
    gradOutsideVecs[outsideWordIdx] = -(1 - sigmoid(uo_vc)) * centerWordVec

    ### Use sua implementação da função sigmoid aqui.

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord,
             windowSize,
             outsideWords,
             word2Ind,
             centerWordVectors,
             outsideVectors,
             dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Modelo de skip-gram no word2vec

    Implemente o modelo skip-gram nesta função.

    Argumentos:
    currentCenterWord - string da palavra central atual
    windowSize - inteiro, tamanho da janela de contexto
    outsideWords - lista de não mais do que 2 * strings windowSize, as palavras externas
    word2Ind - um objeto dict que mapeia palavras para seus índices 
               na lista de vetores de palavras
    centerWordVectors - matriz dos vetores da palavra central (como linhas) com shape
                        (num palavras no vocabulário, comprimento do vetor da palavra)
                        para todas as palavras do vocabulário ( V no enunciado pdf)
    outsideVectors - matriz dos vetores externos (como linhas) com shape
                        (num palavras no vocabulário, comprimento do vetor da palavra)
                        para todas as palavras do vocabulário (U no enunciado pdf)
    word2vecLossAndGradient - a função de custo e gradiente para
                               um vetor de predição dado os vetores de palavra outsideWordIdx,
                               poderia ser um dos dois funções de perda que você implementou acima.

    Retorna:
    loss - o valor da função de custo para o modelo skipgrama de (J no enunciado pdf)
    gradCenterVec - o gradiente em relação ao vetor da palavra central
                     com shape (comprimento do vetor da palavra,)
                     (dJ / dV no enunciado pdf)
    gradOutsideVectors - o gradiente em relação a todos os vetores de palavras externos
                    com shape (num palavras no vocabulário, comprimento do vetor da palavra)
                    (dJ / dU  no enunciado pdf)
                                        
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### Seu código aqui (~8 Lines)

    centerWordVec = centerWordVectors[word2Ind[currentCenterWord]]

    for o in outsideWords:
        outsideWordIdx = word2Ind[o]
        tmp_loss, tmp_gradCenterVecs, tmp_gradOutsideVectors = word2vecLossAndGradient(centerWordVec, 
                outsideWordIdx, outsideVectors,dataset)

        loss += tmp_loss
        gradCenterVecs += tmp_gradCenterVecs
        gradOutsideVectors += tmp_gradOutsideVectors

    gradCenterVecs_tmp = np.zeros(gradCenterVecs.shape)
    gradCenterVecs_tmp[ word2Ind[currentCenterWord] ] = gradCenterVecs[ word2Ind[currentCenterWord] ]
    gradCenterVecs = gradCenterVecs_tmp


    ### Seu código acaba aqui

    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# A seguir, funções de teste. NÃO MODIFIQUE #
#############################################


def word2vec_sgd_wrapper(word2vecModel,
                         word2Ind,
                         wordVectors,
                         dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
#    print_debug(f"*" * 100)
#    print_debug(f"*" * 100)
#    print_debug(f"*" * 100)
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(centerWord, windowSize1, context,
                                     word2Ind, centerWordVectors,
                                     outsideVectors, dataset,
                                     word2vecLossAndGradient)
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print(
        "==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print(
        "==== Gradient check for skip-gram with negSamplingLossAndGradient ===="
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset,
                                         5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset,
                       negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()
