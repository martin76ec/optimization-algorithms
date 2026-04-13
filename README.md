
# Optimization Algorithms Benchmarks


## Code

1. the block code is similar for all the optimizers
2. the optimizers code are the provided by the class (from d2l resources)

```
@square_decorator(title="PSO", width=90)
def pso_benchmark():
    result_set = {}

    for group, params in PSO_PARAMS.items():
        best_sol, best_val, _ = particle_swarm_opt(
            f=params["function"],
            p_num=params["p_num"],
            n=params["n"],
            bounds=params["bounds"],
            max_epochs=params["max_epochs"],
        )
        result_set[group] = [best_sol, best_val]

    params = pandas.DataFrame.from_dict(SA_PARAMS, orient="index")

    results = pandas.DataFrame.from_dict(
        result_set, orient="index", columns=["best_solution", "best_value"]
    )

    print(results)
```

## PSO

### Params

```
                   p_num  n max_epochs alph1 alph2
ackley_fast           30  2        100   1.5   1.5
ackley_standard       50  2        300   2.0   2.0
ackley_robust        100  2        500   2.0   2.0

goldstein_fast        40  2        150   1.8   1.8
goldstein_standard    80  2        400   2.0   2.0
goldstein_robust     150  2        800   2.0   2.0

easom_fast           100  2        200   1.5   2.5
easom_standard       500  2        500   2.0   2.0
easom_robust        1000  2       1000   2.0   2.0

```

### Results


```
                                                    best_solution  best_value
ackley_fast         [0.12317583961444312, -0.0073188115036062484]    0.708054
ackley_standard     [-0.002098223624808937, -0.00301746839749506]    0.010755
ackley_robust       [0.003932605381034904, 0.0005379309624169437]    0.011646

goldstein_fast        [0.001354801619513124, -1.0004289678792895]    3.000669
goldstein_standard  [0.00010038142052584687, -1.0000469482475363]    3.000005
goldstein_robust    [-0.0018188628604821622, -1.0009197093766924]    3.000839

easom_fast               [3.1592012860791447, 3.1490231452347057]   -0.999452
easom_standard             [3.141689248018699, 3.141441194362356]   -1.000000
easom_robust             [3.1437717834299743, 3.1424482145712487]   -0.999992
```

1. Higher coeficients improves the accuracy
2. It is not able to reach the global min for ackley and goldstein, but gets pretty close

## Simulated Annealing

### Params

```
                     t_init  cooling_rate  epochs
ackley_fast              50         0.850    1000
ackley_standard         100         0.950    5000
ackley_robust           200         0.990   20000

goldstein_fast        10000         0.800    2000
goldstein_standard  1000000         0.900   10000
goldstein_robust    5000000         0.980   50000

easom_fast                5         0.900    5000
easom_standard           10         0.990   50000
easom_robust             20         0.999  200000

```

### Results

```
                                                      best_solution  best_value
ackley_fast                [-17.015598663347266, 30.99788772805747]   19.871983
ackley_standard      [-0.007478777523796776, -0.005996481062267023]    0.029558
ackley_robust        [0.007622052850944527, -0.0038063851092566825]    0.026029

goldstein_fast       [-0.0067361656593438735, 0.020521391450170956]    0.073468
goldstein_standard     [-0.008594079102659746, 0.01571313275325692]    0.059178
goldstein_robust      [0.00297583372521415, -0.0017069783024328267]    0.010017

easom_fast                   [3.99258527646003, -69.00998336273852]   20.003008
easom_standard           [-62.003084318446895, -36.001607660529686]   19.999535
easom_robust        [0.0018164272310396967, -0.0033252609007288214]    0.011099
```


1. The faster it gets cold the worse for tricky functions with many local minima
2. Any of the configurations was able to reach goldstein global minima, but a lower initial temp helps to get closer to the target

## Genetic Algorithm

### Params

```
                    population_size  chromosome_len  max_epochs  n_elites  mut_prob
ackley_fast                      50              24         100         1      0.02
ackley_standard                 150              32         250         2      0.05
ackley_robust                   300              40         500         5      0.10

goldstein_fast                  100              30         150         2      0.01
goldstein_standard              250              40         300         5      0.02
goldstein_robust                500              60         600        10      0.05

easom_fast                      200              32         200         1      0.05
easom_standard                  600              32         500         2      0.10
easom_robust                   1500              40        1000         2      0.20

```

### results

```
                                                        best_solution  best_value
ackley_fast             [0.008001953601954881, -0.008001953601954881]    0.035415
ackley_standard      [-0.0005000076295118561, -0.0005000076295118561]    0.002013
ackley_robust       [-0.00028125026822323207, -0.00028125026822323...    0.001129

goldstein_fast           [-6.103701895199265e-05, -0.999969481490524]    3.000002
goldstein_standard      [1.9073504518019035e-06, -0.9999990463247741]    3.000000
goldstein_robust         [1.862645149230957e-09, -1.0000000027939677]    3.000000

easom_fast                   [3.1235217822537464, 3.1418326085298105]   -0.999510
easom_standard                [3.135728999771132, 3.1418326085298105]   -0.999948
easom_robust                 [3.1462699377727006, 3.1514197839925657]   -0.999822
```

1. It is considerably slower than Simulated Annealing but performs much better
2. Almost each combination reached, or was very close, to the target


# Travling Salesman 

## Fully Connected Nodes

[](./figures/fig1.png)

## Random Nodes (N x N)

[](./figures/fig2.png)

# Neuron
