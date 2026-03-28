from functions.ackley import ackley
from functions.easoms import easom
from functions.gp import goldstein_price

from functions.ackley import ackley
from functions.easoms import easom
from functions.gp import goldstein_price

PSO_PARAMS = {
    # --- ACKLEY FUNCTION ---
    "ackley_fast": {
        "function": ackley,
        "p_num": 30,
        "n": 2,
        "max_epochs": 100,
        "alph1": 1.5,
        "alph2": 1.5,
        "bounds": [-32.768, 32.768],
    },
    "ackley_standard": {
        "function": ackley,
        "p_num": 50,
        "n": 2,
        "max_epochs": 300,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-32.768, 32.768],
    },
    "ackley_robust": {
        "function": ackley,
        "p_num": 100,
        "n": 2,
        "max_epochs": 500,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-32.768, 32.768],
    },
    # --- GOLDSTEIN-PRICE FUNCTION ---
    "goldstein_fast": {
        "function": goldstein_price,
        "p_num": 40,
        "n": 2,
        "max_epochs": 150,
        "alph1": 1.8,
        "alph2": 1.8,
        "bounds": [-2.0, 2.0],
    },
    "goldstein_standard": {
        "function": goldstein_price,
        "p_num": 80,
        "n": 2,
        "max_epochs": 400,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-2.0, 2.0],
    },
    "goldstein_robust": {
        "function": goldstein_price,
        "p_num": 150,
        "n": 2,
        "max_epochs": 800,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-2.0, 2.0],
    },
    # --- EASOM FUNCTION ---
    "easom_fast": {
        "function": easom,
        "p_num": 100,
        "n": 2,
        "max_epochs": 200,
        "alph1": 1.5,
        "alph2": 2.5,
        "bounds": [-100.0, 100.0],
    },
    "easom_standard": {
        "function": easom,
        "p_num": 500,
        "n": 2,
        "max_epochs": 500,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-100.0, 100.0],
    },
    "easom_robust": {
        "function": easom,
        "p_num": 1000,
        "n": 2,
        "max_epochs": 1000,
        "alph1": 2.0,
        "alph2": 2.0,
        "bounds": [-100.0, 100.0],
    },
}
