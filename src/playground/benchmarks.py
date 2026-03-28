from algorithms.ga import genetic_algorithm
from algorithms.pso import particle_swarm_opt
from algorithms.sa import simulated_annealing
from constants.ga import GA_PARAMS
from constants.pso import PSO_PARAMS
from constants.sa import SA_PARAMS
from utils.decorators import square_decorator
import pandas


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


@square_decorator(title="Simulated Annealing", width=90)
def sa_benchmark():
    result_set = {}

    for group, params in SA_PARAMS.items():
        best_sol, best_val, _ = simulated_annealing(
            func=params["function"],
            bounds=params["bounds"],
            max_iter=params["epochs"],
            initial_temp=params["t_init"],
            cooling_rate=params["cooling_rate"],
        )
        result_set[group] = [best_sol, best_val]

    params = pandas.DataFrame.from_dict(SA_PARAMS, orient="index")

    results = pandas.DataFrame.from_dict(
        result_set, orient="index", columns=["best_solution", "best_value"]
    )

    print(results)


@square_decorator(title="Genetic algorithm", width=90)
def ga_benchmark():
    result_set = {}

    for group, params in GA_PARAMS.items():
        coords, final_eval, _ = genetic_algorithm(
            params["function"],
            params["population_size"],
            params["chromosome_len"],
            params["max_epochs"],
            params["n_elites"],
            params["mut_prob"],
            params["bounds"],
        )
        result_set[group] = [coords, final_eval]

    results = pandas.DataFrame.from_dict(
        result_set, orient="index", columns=["best_solution", "best_value"]
    )

    print(results)


def params_show():
    pso = pandas.DataFrame(data=PSO_PARAMS)
    sa = pandas.DataFrame(data=SA_PARAMS)
    ga = pandas.DataFrame(data=GA_PARAMS)

    print(pso.T.drop(columns=["function", "bounds"]))


def benchmarks_run():
    pso_benchmark()
    print("\n\n")
    sa_benchmark()
    print("\n\n")
    ga_benchmark()
