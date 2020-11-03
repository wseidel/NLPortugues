# -*- coding: utf-8 -*-


# https://keras.io/examples/nlp/text_classification_with_transformer/
import tensorflow as tf
# tf.__version__
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.util import *
import nltk

import argparse

my_parser = argparse.ArgumentParser()

# my_parser.add_argument('--etapa', choices=['pre','treino','teste'], default=False, required=True)
my_parser.add_argument('--b2w_path', action='store', type=str, required=True,
            help='Caminho para o arquivo cotendo dados originais: B2W-Reviews01.csv')
my_parser.add_argument('--lstm_bidirect', action='store_true', default=False,
            help='Utiliza a camada LSTM Bidirecional ao invés da simples. Default: desligado')
my_parser.add_argument('--nilc_path', action='store', default=False,
            help='Caminho para o arquivo word2vec do NILC. Se não especificado, será usado embedding vazio.')
my_parser.add_argument('--dropout', action='store', type=float, default=0.0,
            help='Valor em float da camada de Dropout. Default: 0. Exemplo: --dropout 0.5')
my_parser.add_argument('--tammax_sentence', action='store', type=int, default=40,
            help='Tamanho máximo da sentença a ser usada. Default: 40')
my_parser.add_argument('--vocab_size', action='store', type=int, default=20000,
            help='Tamanho do Vocabulário. Default: 20.000')
my_parser.add_argument('--epocas', action='store', type=int, default=100,
            help='Quantidade de épocas. Default: 100')

args = my_parser.parse_args()

TAMMAX_SENTENCE = args.tammax_sentence
VOCAB_SIZE = args.vocab_size
EH_BIDIRECIONAL = args.lstm_bidirect
DROPOUT = args.dropout
EMBED_DIM = 50
QNT_EPOCAS_TREINO = args.epocas
TRAIN_SIZE=0.75
TEST_SIZE=0.1
PRINT_SPLIT_SIZE=True

EH_EMBEDDING_NILC = False
NILC_W2V_DATAFILE = ""
if args.nilc_path:
  EH_EMBEDDING_NILC = True
  NILC_W2V_DATAFILE = args.nilc_path


B2W_DATAFILE = args.b2w_path
# B2W_DATAFILE = "/home/wseidel/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv"
# B2W_DATAFILE = "/home/wseidel/workspaces/usp/b2w-reviews01/B2W-10k.csv"

# Path para o arquivo de dados de embeddings do NILC
# NILC_W2V_DATAFILE = "/home/wseidel/workspaces/usp/NILC/word2vec_200k.txt"
# NILC_W2V_DATAFILE = "/home/wesley/workspaces/usp/data/nilc/word2vec_200k.txt"



print("Configurações:")
# print(" - ETAPA.....:", args.etapa )
print(" - B2W.......:", B2W_DATAFILE )

print("Configurações:")
print(" - TAMMAX_SENTENCE.....:", TAMMAX_SENTENCE )
print(" - VOCAB_SIZE..........:", VOCAB_SIZE )
print(" - EH_BIDIRECIONAL.....:", EH_BIDIRECIONAL )
print(" - DROPOUT.............:", DROPOUT )
print(" - EMBED_DIM...........:", EMBED_DIM )
print(" - QNT_EPOCAS_TREINO...:", QNT_EPOCAS_TREINO )
print(" - TRAIN_SIZE..........:", TRAIN_SIZE)
print(" - TEST_SIZE...........:", TEST_SIZE)
print(" - EH_EMBEDDING_NILC...:", EH_EMBEDDING_NILC )
print(" - NILC_W2V_DATAFILE...:", NILC_W2V_DATAFILE )
print("------------------------")




# Path para o arquivo de dados da b2w
# B2W_DATAFILE = "/home/wseidel/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv"
# B2W_DATAFILE = "/home/wseidel/workspaces/usp/b2w-reviews01/B2W-10k.csv"

# Path para o arquivo de dados de embeddings do NILC
# NILC_W2V_DATAFILE = "/home/wseidel/workspaces/usp/NILC/word2vec_200k.txt"
# NILC_W2V_DATAFILE = "/home/wesley/workspaces/usp/data/nilc/word2vec_200k.txt"


# nltk.download('stopwords')


# Carregar dados a serem analisados
print("[Lendo os dados originais...]")
b2wCorpus = pd.read_csv(B2W_DATAFILE, sep=';', usecols=["review_text", "overall_rating"])

print("[Pre-processando...]")
df_to_work = preprocess(b2wCorpus)

print("[Realizando o split...]")
train, test, val = train_test_val_split(df_to_work, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

print("[Ordenando os dados de treino pelo tamanho das sentencas...]")
sort_by_size(train, 'review_text')

print("[Separando X e Y...]")
x_train, y_train = getXY(train['review_text'], train['overall_rating'])
x_test,  y_test  = getXY(test['review_text'], test['overall_rating'])
x_val,   y_val   = getXY(val['review_text'], val['overall_rating'])


if PRINT_SPLIT_SIZE:
    print("[Sobre o Split]") 
    print("--train..:", len(train), round(len(train) / len(df_to_work),3) ) 
    print("--test...:", len(test), round(len(test) / len(df_to_work),3) )
    print("--val....:", len(val), round(len(val) / len(df_to_work),3) )
    print("--" * 20) 
    # print("x_train..:", len(x_train), "Last value(Y,X)..: [", y_train[-1], "] ", x_train[-1] )
    # print("x_train..:", len(x_train), "Last value(Y,X)..: [", y_train[-10], "] ", x_train[-10] )
    # print("x_test...:", len(x_test),  "Last value(Y,X)..: [", y_test[-1], "] ", x_test[-1] )
    # print("x_val....:", len(x_val),  "Last value(Y,X)..: [", y_val[-1], "] ", x_val[-1] )



print("[Preparando camada de vetorização...]")
vectorize_layer = get_vectorize_layer_adapted( x_train, VOCAB_SIZE, TAMMAX_SENTENCE)


print("[Preparando camada Embedding...]")
embedding_layer = layers.Embedding(VOCAB_SIZE+1, output_dim=EMBED_DIM, input_length=TAMMAX_SENTENCE, name="Emb_Zerado")
if EH_EMBEDDING_NILC:
    print("[Preparando camadam Embedding da NILC..]")
    embedding_layer = get_nilc_embedding_layer(NILC_W2V_DATAFILE, vectorize_layer)

model = create_model( vectorize_layer=vectorize_layer,
                      embed_layer=embedding_layer,
                      is_birect=EH_BIDIRECIONAL,
                      vocab_size=VOCAB_SIZE, 
                      dropout_prob=DROPOUT 
                      )

model.compile( "adam",
               "sparse_categorical_crossentropy", 
               metrics=["accuracy"]
               )


model.summary()


model_name = "Tam_Sentenca_" + str(TAMMAX_SENTENCE)
model_name += '_Rede:_'+''.join([ l.name + '_' for l in  model.layers ])

create_dir_if_not_exists('saved_models')
checkpoint_path = 'saved_models/' + model_name + ".ckpt" 

es = tf.keras.callbacks.EarlyStopping(  monitor='val_loss', 
                                        mode='min', 
                                        verbose=1, 
                                        patience=5,
                                        restore_best_weights=True
                                        )

cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        monitor='val_loss',    
                                        mode='min',
                                        verbose=1,
                                        save_weights_only=True,
                                        save_best_only=True
                                        )

history = model.fit(x_train, y_train,
                    batch_size=32, 
                    epochs=QNT_EPOCAS_TREINO,
                    validation_data=(x_val, y_val),
                    callbacks=[es, cp]
                    )


print("[Restaurando o modelo do fim da melhor época...]")
model.load_weights(checkpoint_path)

loss, accuracy = model.evaluate(x=x_test,y=y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



create_dir_if_not_exists('saved_figs')
text = "Tam_Sentenca:" + str(TAMMAX_SENTENCE)
text += '\nRede:\n '+''.join([ l.name + '\n ' for l in  model.layers ])
filename_acc  = text.replace("\n","_").replace(" ","") + "_Accuracy.jpg"
filename_loss = text.replace("\n","_").replace(" ","") + "_Loss.jpg"

renderfig( 'saved_figs/' + filename_acc, acc , val_acc , "Treino", "Validação", "Acurácia do Treino e Validação", "Épocas", "Acurácia", text )
renderfig( 'saved_figs/' + filename_loss, loss, val_loss, "Treino", "Validação", "Loss do Treino e Validação", "Épocas", "Acurácia", text )

