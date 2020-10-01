https://explained.ai/matrix-calculus/

a.1) Primeiro, implemente o método sigmoid, que recebe um vetor e aplica a função
sigmóide a ele.  - OK

a.2) Em seguida, implemente 
a.2.a) o custo e o gradiente softmax no método naiveSoftmaxLossAndGradient e  -- ACHO Q TA OK
a.2.b) o custo da amostragem negativa e gradiente no método negSamplingLossAndGradient 

a.3) Finalmente, preencha a implementação para o modelo skip-gram no método skipgram. -- ACHO Q TA OK

a.4) Quando terminar, teste sua implementação executando python word2vec.py.



   currentCenterWord.: c
   windowSize........:   3
   outsideWords......: ['a', 'b', 'e', 'd', 'b', 'c']
   word2Ind..........: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

   centerWordVectors.: 
[[-0.96735714 -0.02182641  0.25247529]
 [ 0.73663029 -0.48088687 -0.47552459]
 [-0.27323645  0.12538062  0.95374082]
 [-0.56713774 -0.27178229 -0.77748902]
 [-0.59609459  0.7795666   0.19221644]]

   outsideVectors....: 
[[-0.6831809  -0.04200519  0.72904007]
 [ 0.18289107  0.76098587 -0.62245591]
 [-0.61517874  0.5147624  -0.59713884]
 [-0.33867074 -0.80966534 -0.47931635]
 [-0.52629529 -0.78190408  0.33412466]]
