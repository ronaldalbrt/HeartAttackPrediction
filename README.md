<h1 align="center">
<br> Heart Attack probability estimation with groupable and low-dimensional data.
</h1>
Repository for the course on Computational Intelligence II at  <a href="https://www.cos.ufrj.br/" > PESC - Programa de Engenharia de Sistemas e Computação</a> from <a href="https://ufrj.br/" >UFRJ - Federal University of Rio de Janeiro</a>, ministered by <a href="http://lattes.cnpq.br/2718664296804955">Prof.  Carlos Eduardo Pedreira</a>.

Developed by Ronald Albert.
<h2 align="center">
The project
</h2>
The project consists of a study on the estimation of the probability of a heart attack given a set of features. The data provided is low-dimensional and groupable, therefore the approach taken is consitent with the data provided. 

It's entirely implemented in Julia, and the results are available in the results folder. In order replicate the results, one must install julia and run the following command.
```
julia test/test.jl
```

<h2 align="center">
File list
</h2>
<ul>
    <li><h3>src/HeartAttackPrediction.jl</h3></li>
    <p>Main module of the project where every other module is imported.</p>
    <li><h3>src/ModelEstimation.jl</h3></li>
    <p>Module for the estimation and prediction with the proposed models. Such module is composed of two functions train_model and test_model.</p>
    <li><h3>SubsetSelection.jl</h3></li>
    <p>Module for implementing the Best Subset Selection procedure.</p>
</ul>

