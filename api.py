from fastapi import FastAPI
import numpy as np
from scipy.stats import norm, gmean
N_prime = norm.pdf
N = norm.cdf

import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()

import pandas as pd
import cython
import math as m

app = FastAPI()

@app.get("/euro_vanilla_call")
def euro_vanilla_call(spot_price, strike_price, time_to_maturity, interest_rate, volatility_of_underlying_asset):
    S = float(spot_price)
    K = float(strike_price)
    T = float(time_to_maturity)
    r = float(interest_rate)
    sigma = float(volatility_of_underlying_asset)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    
    return call


@app.get("/imp_vol")
def imp_vol(taxa, media_geometrica_dos_precos, preco_de_exercicio_da_opcao, observed_price, qnt_periodos):
    volatility_candidates = np.arange(0.01,4,0.0001)
    price_differences = np.zeros_like(volatility_candidates)
    S = float(media_geometrica_dos_precos)
    K = float(preco_de_exercicio_da_opcao)
    observed_price = float(observed_price)
    r = float(taxa)
    T = float(qnt_periodos)

    for i in range(len(volatility_candidates)):
        candidate = volatility_candidates[i]
        price_differences[i] = observed_price - black_scholes_call(S, K , T, r, candidate)
        
    idx = np.argmin(abs(price_differences))
    implied_volatility = volatility_candidates[idx]
    
    sigma = implied_volatility
    price = black_scholes_call(S, K, T, r, sigma)

    imp_vol = implied_volatility_call(observed_price, S, K, T, r)
        
    return {"imp_vol": imp_vol}

@app.get("/opcao_exotica")
def opcao_exotica(Current_price_of_underlying_asset, Volatility, Strike_price, Risk_free_rate, Time_to_maturity, Time_intervals, flag_call_or_put):

    S0 = float(Current_price_of_underlying_asset)
    sigma = float(Volatility)
    K = float(Strike_price)
    r = float(Risk_free_rate)
    T = float(Time_to_maturity)
    Nt = float(Time_intervals)
    flag = flag_call_or_put
    if flag != 'call' and flag != 'put':
        return "campo flag incorreto (ecolha entre 'call' e 'put')"

    adj_sigma=sigma*m.sqrt((2*Nt+1)/(6*(Nt+1)))
    rho=0.5*(r-(sigma**2)*0.5+adj_sigma**2)
    d1 = (m.log(S0/K)+(rho+0.5*adj_sigma**2)*T)/(adj_sigma*m.sqrt(T))
    d2 = (m.log(S0/K)+(rho-0.5*adj_sigma**2)*T)/(adj_sigma*m.sqrt(T))
    if (flag=="call"):
        return m.exp(-r*T)*(S0*m.exp(rho*T)*norm.cdf(d1)-K*norm.cdf(d2)) # Important to note that BSM has CDF
    elif (flag =="put"):
        return m.exp(-r*T)*(K*norm.cdf(-d2)-S0*m.exp(rho*T)*norm.cdf(-d1))
    


def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call
def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    
    vega = S  * np.sqrt(T) * N_prime(d1)
    return vega
def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.1

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma