# 1. The project : mapReduceLda
This project aims to impelement a parallel version of the Latent Dirichlet Allocation algorithm which is one of the most used topic modeling algorithm in the industry. For theoretcial details, please take a look on the [papers](https://github.com/neroksi/mapReduceLda/tree/master/papers). 

# 2. The model
The core idea consists in running many concurrent Gibbs samplers. Each gibbs sampler compute its own `documents-counts` and `words-counts` matrices. At the end of each step, the `words-counts` matrix of each sampler is communicated to the master which is in charge of updating of the whole set of matrices. This is repeated until convergence ...

# 3. Our implementation
