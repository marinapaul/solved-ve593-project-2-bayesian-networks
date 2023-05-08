Download Link: https://assignmentchef.com/product/solved-ve593-project-2-bayesian-networks
<br>
<strong>Abstract</strong>

The goal of this project is to help you better understand Bayesian networks and learn to how to apply them in practice.

<h1>Introduction</h1>

Given some data, in order to use a Bayesian network (BN) to answer some inference queries about it, one would need to go through the following steps:

<ul>

 <li>Learn the structure of a BN from data,</li>

 <li>Learn the parameters of the BN obtained in the previous step,</li>

 <li>Use the resulting fully-specified BN for inference.</li>

</ul>

In the first part of this project, you will implement some of the algorithms covered in class for those three steps, namely:

<ul>

 <li>the K2 algorithm with different score functions for the first step,</li>

 <li>the maximum likelihood approach for the second step,</li>

 <li>the variable elimination algorithm for the last step.</li>

</ul>

In the second part of this project, you will use your implementation of those algorithms to study two datasets that we provide: wine and protein. For the first dataset, your learned BN will be used to predict the quality of wines and in the second, it will be used to evaluate the risk of a patient for a disease given the presence of some amino acid types in her DNA sequence.

<h1>Part 1: Learning a Bayes Net from Data</h1>

In this part, we assume that the training data denoted <em>X </em>where <em>N </em>is the number of training points and <em>n </em>is the number of variables is represented in Python as a 2-dimensional list (i.e., list of lists) data=[[<em>x</em><sub>11</sub><em>,x</em><sub>12</sub><em>,…,x</em><sub>1<em>n</em></sub>]<em>,</em>[<em>x</em><sub>21</sub><em>,x</em><sub>22</sub><em>,…,x</em><sub>2<em>n</em></sub>]<em>,…,</em>[<em>x<sub>N</sub></em><sub>1</sub><em>,x<sub>N</sub></em><sub>2</sub><em>,…,x<sub>Nn</sub></em>]].

<h2>Structure Learning</h2>

Recall that learning the structure of a BN amounts to solving the following optimization problem

max<em>f</em>(<em>G,</em><em>X</em>)

<em>G</em>∈G

where G= all DAGs with <em>n </em>nodes and <em>f</em>(<em>G,</em><em>X</em>) is a score function that evaluates how <em>G </em>fits the data <em>X</em>. Given the difficulty of this problem, simplification assumptions have to be made (e.g., so that <em>f </em>can be decomposed as a sum of scores over nodes) and heuristic methods are usually employed (e.g., so that the size of the search space is reduced).

For solving this problem, you will implement the K2 algorithm, which has been presented in class. Recall this algorithm is an heuristic method for solving the following simpler optimization problem:

max<em>f</em>(<em>G,</em><em>X</em>)

<em>G</em>∈G<sup>0</sup>

where G<sup>0</sup>=DAGs satisfying a fixed topological order, which we assume is the order of the variables in which they appear in <em>X</em>. This is without loss of generality because you can always reorder the columns of <em>X </em>before running the K2 algorithm. You will make sure that your implementation is generic such that it can be used with different score functions. For this reason, your K2 algorithm should respect the following signature:

[graph, totalScore] = K2Algorithm(K, data, scoreFunction) where the parameters are defined as follows:

<ul>

 <li>K is the maximum number of allowed parents for a variable,</li>

 <li>data is a list of lists, containing the training data.,</li>

 <li>scoreFunction is a score function whose signature is: score = scoreFunction(variable, parents, data)</li>

</ul>

where variable is the index of a variable (i.e., a column in <em>X</em>), parents is a list of indices that correspond to potential parents for variable, data is the training data and score is the score obtained if parents were connected to variable in the BN. See the explanation below for totalScore to understand the relation between score function <em>f </em>and scoreFunction.

<ul>

 <li>graph represents the resulting BN structure. For simplicity, we assume that it is encoded as an adjacency matrix <em>G</em></li>

</ul>

{0<em>,</em>1}<em><sup>n</sup></em><sup>×<em>n </em></sup>where <em>g<sub>ij </sub></em>= 1 indicates the existence of a directed edge from node <em>i </em>to node <em>j </em>and <em>g<sub>ij </sub></em>= 0 indicates the non-existence of such an edge. In Python, graph is a 2-dimensional list. See the example below:

graph = [[0, 1, 1, 0] [0, 0, 0, 0]

[0, 1, 0, 1]

[0, 0, 0, 0]]

It means that variable 1 is the parent of variables 2 and 3, variable 3 is the parent of variables 2 and 4.

<ul>

 <li>totalScore represents <em>f</em>(<em>G,</em><em>X</em>) that is the sum of the scores given by scoreFunction for each variable if the BN structure <em>G </em>is given by graph. In other words, it corresponds to the value of the objective function that has been maximized by the K2 algorithm and graph corresponds to the graph structure that attains that value.</li>

</ul>

For the score functions, you will implement the K2 original score and the Bayesian Information Criterion (BIC) score (as described in class) according to the signature of scoreFunction. For convenience, we provide you the formula of the <em>i</em>-th variable, denoted <em>X<sub>i</sub></em>, for any <em>i </em>= 1<em>,…,n</em>. The Bayesian K2 score function is given by:

!

where <em>π<sub>i </sub></em>is the list of candidate parents of <em>X<sub>i</sub></em>, <em>q<sub>i </sub></em>is the number of possible assignments for those parents, <em>r<sub>i </sub></em>is the number of possible values for <em>X<sub>i</sub></em>, <em>N<sub>ijk </sub></em>is the number of simultaneous occurrences of the <em>k</em>-th value for <em>X<sub>i </sub></em>and the <em>j</em>-th assignment of its parents, and <em>N<sub>ij </sub></em>= <sup>P</sup><em><sub>k </sub>N<sub>ijk</sub></em>. In practice, to avoid overflow and underflow issues, you will use and compute the log of this score. Typically, a higher score means the structure fits better the training data. However, if the dataset is large, the BN structure learned with the K2 score may be too complex (because adding edges usually increases the likelihood). The Bayesian Information Criterion (BIC) score is applied to avoid this problem. It can be computed as follows for variable <em>X<sub>i</sub></em>:

<h2>Parameter learning</h2>

Assuming that a BN structure is given as an adjacency matrix, its (conditional) probabilities are estimated for each node (i.e., variable) by maximum likelihood. The main function realizing this task has the following signature: cpt = MLEstimationVariable(variable, parents, data) where the arguments have the same semantics as previously and cpt is a (<em>d </em>+ 1)-dimensional list with <em>d </em>the number of parents. More specifically, cpt[<em>i</em><sub>0</sub>][<em>i</em><sub>1</sub>]<em>…</em>[<em>i<sub>d</sub></em>] corresponds to the conditional probability that variable takes its <em>i</em><sub>0</sub>-th value given that for <em>j </em>= 1<em>,…,d</em>, its <em>j</em>-th parent takes its <em>i<sub>j</sub></em>-th value.

To learn the whole BN model, you will call MLEstimation on each variable given the parents indicated by the structure graph obtained in the previous step. This operation should be realized in the following function:

cptList = MLEstimationVariable(graph, data)

where graph is an adjacency matrix and cptList is the list of CPTs, each represented as a list for each variable in the order they appear in data.

<h2>Inference</h2>

With a fully-specified Bayes net, it is possible to perform inference tasks. The answer of inference queries can be computed using the variable elimination algorithm that you will implement. It should have the following signature:

prob = variableElimination(index, observations, model) where

<ul>

 <li>index is the index of the hidden variable of interest,</li>

 <li>observations is a list of pairs (<em>i,o</em>) where <em>i </em>is the index of an observed variable and <em>o </em>is the index of the corresponding observed value,</li>

 <li>model is a pair whose first component is an adjacency matrix and whose second component is a list of CPTs (each represented as a list as described previously),</li>

 <li>prob is a list containing the resulting probabilities.</li>

</ul>

For instance, the call of variableElimination(4, [(1, 1), (3, 2)], M) computes P(<em>X</em><sub>4 </sub>|<em>X</em><sub>1 </sub>= <em>x</em><sub>11</sub><em>,X</em><sub>3 </sub>= <em>x</em><sub>32</sub>) assuming that the BN is represented by model M, the variables are named <em>X<sub>i</sub></em>’s and their possible values are denoted <em>x<sub>ij</sub></em>’s.

<h2>Coding Tasks</h2>

To summarize, for Part 1, you need to code the following functions:

<ul>

 <li>K2Algorithm</li>

 <li>K2Score and BICScore according to the specification of scoreFunction</li>

 <li>MLEstimationVariable and MLEstimation</li>

 <li>variableElimination</li>

</ul>

The functions related to structure learning and parameter learning should be provided in a single file named structurelearning.py. The function variableElimination should be provided in a file named variableelimination.py.

For testing, you can run/debug your code on a small artificial dataset you create and/or the example given on Slide 15 on BN learning.

Bonus points will be given if you implement some of the other algorithms seen in class (i.e., the heuristic method based on the maximum weight spanning tree<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or the belief propagation algorithm) and experimentally compare them to the previous methods in your report. In that case, please name your source code as algorithm.py where you replace algorithm by the name of the algorithm you implemented.

<h1>Part 2: Applications</h1>

To help you evaluate your implementations, we provide you some real datasets that have already been cleaned and pre-processed.

<h2>Data Format</h2>

All data in this project are provided as csv files. An example of a dataset with <em>n </em>variables and <em>N </em>cases is in the following format:

The first row of the csv file corresponds to the names of the <em>n </em>variables of the dataset. Each subsequent row contains one data item for the n variables respectively. To read the data from a file named data.csv, you may use the csv module of Python3. For instance:

import csv

with open(’data.csv’, ’r’, encoding=”utf-8″) as csvfile:

reader = csv.reader(csvfile) print(list(reader))

The output should be:

[[var1, var2,…,varn], [d11, d12,…,d1n],…,

[dm1, dm2,…, dmn]]

Before passing the data to your functions, you need to skip the first line of the data file and only store the values of the variables in a list data.

<h2>Datasets</h2>

In the related files of project 2, we provide 2 data sets for you to evaluate your functions: wine.csv and protein.csv.

The file wine.csv contains 12 variables. The first 11 variables describe some features of a wine, which can be experimentally measured:

fixed acidity, volatile acidity, citric acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, ph, sulphets, alcohol. The values for those variables have been normalized and discretized. They can take a discrete value from 1 to 5. The last variable quality is the quality of the wine, also expressed on a discrete scale from 1 to 5. This dataset will be used to understand the important factors that determine the quality of a wine.

The file protein.csv contains 6 variables. The first variable nuc represents the risk for a certain disease. It can take three possible values: -1 means high risk and 0 means normal and 1 means immune. The next 5 variables correspond to amino acid types in a given protein DNA sequence. A protein DNA sequence is shown as Figure 1. There are 4 bases in DNA: adenine (A), cytosine (C), guanine (G), thymine (T). Different sequences of those bases generate different types of amino acids. For instance, the first 3 bases ”TTG” generates a L-Threonine whose abbreviation is ”L”.

Figure 1: A protein sequence.

The risk for the disease (nuc) can be determined by the ratio of each amino acid. The ratios of each amino acid are provided in the 5 last columns (A, R, N, D, and Q) of the dataset. The values have been again normalized and discretized. Those variables can take discrete values from 0 to respectively 5, 6, 6, 1 and 1. This dataset will be used to study which sequence of DNA bases affects most the risk for this disease.

<h2>Coding Tasks</h2>

For learning the BN model, you will only use a part of a dataset (say <em>p</em>%) and do the inference on the other part, denoted D. To determine <em>p</em>, you can choose the number of cases such that the learning process takes over 30 minutes. In the report, you should explain your settings (for the upper bound <em>K </em>of parent nodes, number of cases you use, etc). Please also provide the adjacency matrix Graph and totalScore as the result of your structure learning for each dataset in the report.

To evaluate a BN model, you can compute the answers of some inference queries and compare the results with D. Notably, you will use the following accuracy score for a query P(<em>X</em><sub>1 </sub>|<em>X<sub>i </sub></em>= <em>x<sub>i</sub>,</em>∀<em>i </em>= 2<em>,…,n</em>).

where <em>x </em>= (<em>x</em><sub>1</sub><em>,…,x<sub>n</sub></em>) and <em>ρ</em>(<em>x</em>) = <em>arg </em>max<em><sub>y </sub></em>P(<em>X</em><sub>1 </sub>= <em>y </em>|<em>X<sub>i </sub></em>= <em>x<sub>i</sub>,</em>∀<em>i </em>=

2<em>,…,n</em>), which is the most probable value for <em>X</em><sub>1 </sub>given the values <em>x</em><sub>2</sub><em>,…x<sub>n</sub></em>.

For the wine dataset, the variable of interest, <em>X</em><sub>1</sub>, is taken as quality and for the protein dataset, it is nuc. The score <em>acc </em>will allow you to decide which learning method (i.e., score functions and MSWT if you implement it) is better for this prediction task.

To further investigate the datasets, you can use your inference algorithm to answer other similar queries P(<em>X</em><sub>1 </sub>|<em>X<sub>I </sub></em>= <em>x<sub>I</sub></em>) to find which subset of variables (whose indices are in <em>I </em>⊆ {2<em>,</em>3<em>…,n</em>}) is most useful to predict the variable of interest (i.e., quality or nuc). This is to answer the following question: if we could only observe <em>k &lt; n </em>− 1 variables in <em>X</em><sub>2</sub><em>,…,X<sub>n</sub></em>, which ones are the most related to the variable of interest? You can explain your findings in your report (notably for different <em>k</em>).

The code you write for Part 2 should be saved into two script files wine.py and protein.py.

<h1>Submission and Due Date</h1>

You need to submit a zipped file named P2-Firstname-Lastname.zip where you replace Firstname and Lastname by your first name and last name respectively. This compressed file should contain:

<ul>

 <li>all the source code files mentioned above,</li>

 <li>a txt file explaining how to run your code in order to reproduce your experiments, and</li>

 <li>a <strong>short </strong>report in pdf format.</li>

</ul>

The submission will be on Canvas Assignments. The due date is 11:59 pm on Nov 15th, 2018. There will be a penalty of 20% per day for late submission. Note that submissions will be checked for plagiarism.


