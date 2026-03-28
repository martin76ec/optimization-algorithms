from functions.ackley import ackley
from functions.easoms import easom
from functions.gp import goldstein_price


SA_PARAMS = {
    "ackley_fast": {
        "function": ackley,
        "t_init": 50,
        "cooling_rate": 0.85,
        "epochs": 1000,
        "bounds": [(-32.768, 32.768), (-32.768, 32.768)],
        "target": 0.0
    },
    "ackley_standard": {
        "function": ackley,
        "t_init": 100,
        "cooling_rate": 0.95,
        "epochs": 5000,
        "bounds": [(-32.768, 32.768), (-32.768, 32.768)],
        "target": 0.0
    },
    "ackley_robust": {
        "function": ackley,
        "t_init": 200,
        "cooling_rate": 0.99,
        "epochs": 20000,
        "bounds": [(-32.768, 32.768), (-32.768, 32.768)],
        "target": 0.0
    },
    "goldstein_fast": {
        "function": goldstein_price,
        "t_init": 10000,
        "cooling_rate": 0.80,
        "epochs": 2000,
        "bounds": [(-2.0, 2.0), (-2.0, 2.0)],
        "target": 3.0
    },
    "goldstein_standard": {
        "function": goldstein_price,
        "t_init": 1000000,
        "cooling_rate": 0.90,
        "epochs": 10000,
        "bounds": [(-2.0, 2.0), (-2.0, 2.0)],
        "target": 3.0
    },
    "goldstein_robust": {
        "function": goldstein_price,
        "t_init": 5000000,
        "cooling_rate": 0.98,
        "epochs": 50000,
        "bounds": [(-2.0, 2.0), (-2.0, 2.0)],
        "target": 3.0
    },
    "easom_fast": {
        "function": easom,
        "t_init": 5,
        "cooling_rate": 0.90,
        "epochs": 5000,
        "bounds": [(-100.0, 100.0), (-100.0, 100.0)],
        "target": -1.0
    },
    "easom_standard": {
        "function": easom,
        "t_init": 10,
        "cooling_rate": 0.99,
        "epochs": 50000,
        "bounds": [(-100.0, 100.0), (-100.0, 100.0)],
        "target": -1.0
    },
    "easom_robust": {
        "function": easom,
        "t_init": 20,
        "cooling_rate": 0.999,
        "epochs": 200000,
        "bounds": [(-100.0, 100.0), (-100.0, 100.0)],
        "target": -1.0
    }
}
