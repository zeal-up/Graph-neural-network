# A summary of graph Neural Networks
- [A summary of graph Neural Networks](#a-summary-of-graph-neural-networks)
  - [*Abstract*](#abstract)
  - [Review papers on GNNs](#review-papers-on-gnns)
  - [Spetral methods:](#spetral-methods)
    - [Related theory](#related-theory)
  - [Non-spetral methods:](#non-spetral-methods)
  - [Gated graph neural network](#gated-graph-neural-network)
  - [Attention Graph Neural Network](#attention-graph-neural-network)
  - [General frameworks](#general-frameworks)
  - [Few-shot learning](#few-shot-learning)
  - [Skip connection](#skip-connection)
  - [Source code](#source-code)

## *Abstract*
This summary will try to collect some papers about Graph Neural Networks(GNN) and try
my best to illustrate the evolution of GNN according to my personal understanding. Most of 
this work is based on [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)

## Review papers on GNNs
1. [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
   <br>Most newest survey (Dec 2018). The statement is clearly and logical. Strongly recommended.
2. [Deep Learning on Graphs: A Survey](https://arxiv.org/abs/1812.04202)
3. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
   <br> Pubilshed by DeepMind, Google Brain and other dozens researchers. It proposed a [Graph Network]() (GN) framework. This framework has strong capability to generalize other models(it almost generalize to all kinds of GNN at presents). This paper try to implement GNN as a standard module such as Convolutional Layers in Deep learning. But the paper is difficult to understand and it need sufficient understanding of the present GNN models.

## Spetral methods:
 Convolution operation in the spatial domain is not ideal because of the irregularity of graphs, for the number of neighbors for each node is different. Benefited from graph spectral theory, people try to operate convolutions in graph spectral domain.
 1. The most begining of Graph Neural Network: [The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)(Franco Scarselli et al. 2009)
 2. [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)(Joan Bruna et al. 2014)
 3. [Deep Convolutional Networks on Graph-Structured Data](https://arxiv.org/abs/1506.05163)
   <br> Development of [2], it attempts to make the spectral filters spatially locallized by introducing a parameterization with smooth coefficients.
 4. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)(Defferrard et al. 2017)
   <br> Further simplify [3], Chebyshev polynomial is used to avoid some computational costs.
 5. [Semi-supervised classification with graph convolutional networks.](https://openreview.net/pdf?id=SJU4ayYgl) (Kipf & Welling. 2017)
   <br> A really important work. Many present work will use this framework. The computation is straightly operated on the input sigal, which can be seen as a non-spetral method.

### Related theory
Understanding the evolution of the spetral methods is of significance to master the intrinsic nature of GNN. But the Graph spetral theory is itself a big topic and require sufficient mathematical theory to understand it. So, let go of yourself and skip the confusing part.
1. [Geometric Deep Learning: Going beyond Euclidean data](https://arxiv.org/pdf/1611.08097.pdf)(Bruna et al. 2017)
2. [Graph Signal Processing: Overview, Challenges and Applications](https://arxiv.org/pdf/1712.00468.pdf)(Antonio Ortega et al. 2018)
3. [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/pdf/0912.3848.pdf0)(David K Hammond. 2009)

## Non-spetral methods:
1. [Convolutional networks on graphs for learning molecular fingerprints](https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints0)(Duvenaud et al. NIPS 2015)
   <br> Using different weight matrices for nodes with different degrees.
2. [Diffusion-convolutional neural networks.](https://papers.nips.cc/paper/6212-diffusion-convolutional-neural-networks)(NIPS 2016)
   <br> Diffusion-convolutional neural networks(DCNNs)
3. [Learning convolutional neural networks for graphs](http://proceedings.mlr.press/v48/niepert16.pdf)(2016)
   <br>extracts and normalizes a neighborhood of exactly k nodes for each node. And then the normalized neighborhood serves as the receptive filed for the convolutional operation.
4. [Geometric deep learning on graphs and manifolds using mixture model cnns.](https://arxiv.org/abs/1611.08402)(Monti et al. 2016)
   <br>proposed a spatial-domain model (MoNet) on nonEuclidean domains which could generalize several previous techniques.
5. [Inductive representation learning on large graphs.](https://arxiv.org/abs/1706.02216)(2017)
   <br>proposed the GraphSAGE, a general inductive framework. The framework generates embeddings by sampling and aggregating features from a node’s local neighborhood. `A general inductive framework. Suggest several aggregator functions. Can be extended to gated graph neural network. Also a really important work.`
6. [Fastgcn: fast learning with graph convolutional networks via importance sampling](https://arxiv.org/abs/1801.10247)(2018)

## Gated graph neural network
This basic idea of this kind of GNN is inserting an RNN architecture in the update function of the GNN.
1. [Gated graph sequence neural networks.](https://arxiv.org/abs/1511.05493)(2017)
   <br>proposed the gated graph neural network (GGNN) which uses the Gate Recurrent Units (GRU) in the propagation step, unrolls the recurrence for a fixed number of steps T and uses backpropagation through time in order to compute gradients.
2. [Conversation modeling on reddit using a graph-structured lstm](https://arxiv.org/abs/1704.02080)(2018)
   <br> it is a simplified version since each node in the graph has at most 2 incoming edges (from its parent and sibling predecessor).
3. [Cross-Sentence N-ary Relation Extraction with Graph LSTMs](https://arxiv.org/abs/1708.03743)(2017)
   <br>proposed another variant of the Graph LSTM based on the relation extraction task. The main difference between graphs and trees is that edges of graphs have their labels. And utilizes different weight matrices to represent different labels.
4. [Sentence-state LSTM for text representation](https://arxiv.org/abs/1805.02474)(2018)
   <br>4.proposed the Sentence LSTM (S-LSTM) for improving text encoding. It converts text into graph and utilizes the Graph LSTM to learn the representation. The S-LSTM shows strong representation power in many NLP problems.

## Attention Graph Neural Network
1. [Graph attention networks](https://arxiv.org/abs/1710.10903)(2019)
   <br> The framework if similiar to [gcn kipf & willing](https://openreview.net/pdf?id=SJU4ayYgl), but the adjacency matrix was constructed by an attention mechanism.

## General frameworks
1. [Neural message passing for quantum chemistry.](https://arxiv.org/abs/1704.01212)(2017)
   <br>`A really important work.` The formulation of the general framework is concise and elegant.
   The framework proposed by this paper can generalize 8 GNN models including `spetral and non-spetral methods`.
2. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
3. [Geometric deep learning on graphs and manifolds using mixture model cnns.](https://arxiv.org/abs/1611.08402)(Monti et al. 2016)
   <br>proposed a spatial-domain model (MoNet) on nonEuclidean domains which could generalize several previous techniques.

## Few-shot learning
1. [Zero-shot recognition via semantic embeddings and knowledge graphs](https://arxiv.org/abs/1803.080350)(2018)
2. [Rethinking knowledge graph propagation for zero-shot learning](https://arxiv.org/abs/1805.11724)(2019)
3. [Few-shot learning with graph neural networks](https://arxiv.org/pdf/1711.04043.pdf)(2017)

## Skip connection
1. [Semi-supervised User Geolocation via Graph Convolutional Networks](https://arxiv.org/abs/1804.08049)(2018)
2. [Representation learning on graphs with jumping knowledge networks](https://arxiv.org/abs/1806.03536)(2018)


## Source code 
1. [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)
   <br> It consists of various methods for deep learning on graphs.
2. [Attention Graph network](https://github.com/Diego999/pyGAT)
3. [NPNN](https://github.com/priba/nmp_qc)
