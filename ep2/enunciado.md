# anti-EP2
# Seção Treinamento

### Considere:
 - a) **TAMMAX**: Tamanho máximo de palavras por entrada.
 - b) Numero de palavras não deve exceder TAMMAX (truncar se necessário)
 - c) **batchsize**: Organizar a entrada em batches de tamanho fixo (se usar aleatório)
	 - Potência de 2 (memória é o limitante)
 - d) Cada batch deve ter sentenças do mesmo tamanho
	 - Encontrar a maior sentença e completar as menos com PAD
 - e) Limitar VOCABSIZE em 20.000 palavras (opcional)
	 - As demais palavras serão palavras desconhecidas: UNK

### O treinamento:
- O treinamento se inicia com Embeddind de tamanho fixo. Escolha:
    - EP1
    - NILC (preferencialmente este ou o EP1)
    - Um camada especifica de embedding
- Verificar se a entrada respeita o TAMMAX
    - Truncar se se exceder 	
- **batchsize** (é um hiperparâmetro): Organizar a entrada em batches de tamanho fixo  (se usarmos ordem aleatória)
    - geralmente potencia de 2 (memória é o limitante)
- Cada lote e submetido durante o treinamento a uma rede neural recorrente formada por elementos LSTM.

- Vamos  experimentar  dois  casos.
    - A) Um Encoder unidirecional;
    - B) Um Encoder bidirecional.
- Em ambos os casos, a saıda do Encoder (o codigo) gerado apos toda a entrada ser lida deve ser conectada em uma rede linear densa:
    - O numero de saídas é igual ao numero de classes que desejamos classificar a entrada. (5 no caso)

- No final desta rede inserir uma camada de dropout (para melhorar o desempenho no caso de sobre-ajuste.)

- Experimentos: 
	A) Um Encoder unidirecional: 0% (semdropout), 25% e 50%.
	B) Um Encoder bidirecional:  0%, 25% e 50%.


# Seção Validação 
- Rodar para cada experimento um treinamento que consiste de 50 a 100 epocas.
    - O valor exato deve ser determinada pela deteccao do ponto em que comeca a ver sobreajuste (ou seja, se ja houver aumento no erro do corpus de validacao nas primeiras 50 epocas nao e necessario continuar o treinamento)

	- A cada cinco epocas, avaliar:
		- O erro computado no corpus de validacao.
		- O valor da acurácia do Corpus de validacao.
	- Armazenar estes dados para gerar um grafico como o da Figura 1;
	- Armazenar o erro (loss) do corpus de treinamento e verificar que este sempre diminui.
	- Armazenar os parâmetros do modelo a cada vez que se faca uma rotina de validacao;

