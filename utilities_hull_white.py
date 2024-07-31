import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# load api key from a local .env file
import os
from dotenv import load_dotenv
load_dotenv()

from fredapi import Fred
# documentation: https://pypi.org/project/fredapi/

fred = Fred(api_key=os.getenv('fred_key'))

# assumptions
k = 0.5
sigma = 0.005


def get_spot_curve(sample_start: str = "1/1/2023") -> pd.DataFrame:
    """obtain the US Treasury yield curves from FRED with monthly frequency.

    Args:
        sample_start (str, optional): start date of the sample. Defaults to "1/1/2023".

    Returns:
        pd.DataFrame: yield curves, terms are presented in months
    """
    t_dict = {
        "GS1M": (1 / 12),  # 1 month
        "GS6M": 0.5,  # 6 month
        "GS1": 1,  # 1 year
        "GS2": 2,
        "GS3": 3,
        "GS5": 5,  # 5 year
        "GS7": 7,
        "GS10": 10,  # 10 year
    }

    t_mkt_yld = pd.DataFrame()
    for key in t_dict:
        _m = (
            fred.get_series(key, observation_start=sample_start).to_frame(
                t_dict[key] * 12
            )
            / 100
        )
        t_mkt_yld = t_mkt_yld.merge(_m, left_index=True, right_index=True, how="outer")

    t_mkt_yld = t_mkt_yld.reindex(sorted(t_mkt_yld.columns), axis=1)
    t_mkt_yld.index += pd.offsets.MonthEnd(0)

    return t_mkt_yld


def cs_interpolate(spot_curve: pd.DataFrame) -> pd.DataFrame:
    """cubic spline interpolation to monthly yield rates.

    Args:
        spot_curve (pd.DataFrame): source yield curve from FRED

    Returns:
        pd.DataFrame: interpolated yield curves
    """
    monthly_spot_rate = pd.DataFrame(columns=range(1, 121))
    for _index, _row in spot_curve.iterrows():
        _cs = CubicSpline(_row.index, _row)
        monthly_spot_rate.loc[_index] = _cs(range(1, 121))

    return monthly_spot_rate


def get_inst_fr(spot_curve: pd.Series, delta_t: float = 1 / 12) -> pd.Series:

    # be careful when delta_t is used as the index to retrieve data from a series indexed by month,
    # np.round(delta_t * 12) is used to make sure it would be a integer
    instantaneous_fr = pd.Series()
    instantaneous_fr.loc[0] = spot_curve[1]
    for _m in range(1, len(spot_curve) - 1):
        instantaneous_fr.loc[_m] = (
            spot_curve[_m + np.round(delta_t * 12)]
            + (_m / 12)
            * (spot_curve[_m + np.round(delta_t * 12)] - spot_curve[_m])
            / delta_t
        )
    return instantaneous_fr


def get_theta(
    instantaneous_fr: pd.Series,
    k: float,
    sigma: float,
    delta_t: float = 1 / 12,
) -> pd.Series:
    """calculate theta given the pravailing instantaneous forward curve.

    Args:
        instantaneous_fr (pd.Series): the pravailing instantaneous forward rate.
        k (float): speed of mean reversion
        sigma (float): the standard deviation of rate of change in short rate
        delta_t (float, optional): simulation intervals in years. Defaults to 1/12.

    Returns:
        pd.Series: the average direction in which r moves
    """

    theta_t = pd.Series()
    for _t in range(1, len(instantaneous_fr) - 1):
        theta_t.loc[_t] = (
            (instantaneous_fr[_t + np.round(delta_t * 12)] - instantaneous_fr[_t]) / delta_t
            + k * instantaneous_fr[_t]
            + (sigma**2 / (2 * k)) * (1 - np.exp(-2 * k * _t / 12))
        )

    return theta_t


def get_short_rate(
    spot_curve: pd.Series,
    theta_t: pd.Series,
    k: float,
    sigma: float,
    delta_t: float = 1 / 12,
) -> pd.Series:
    """this function derive the simulated short rate path with Hull-White Model given theta(t), k and sigma.

    Args:
        spot_curve (pd.Series): the observed spot yield curve
        theta_t (pd.Series): the average direction in which r moves, derived from today's zero coupon yield curve
        k (float): speed of mean reversion
        sigma (float): the standard deviation of rate of change in short rate
        delta_t (float, optional): simulation intervals in years. Defaults to 1/12.

    Returns:
        pd.Series: simulated short rate path.
    """

    short_rate = pd.Series()
    # assume r(0) equals to the current spot rate with the shortest term, i.e., one month
    short_rate.loc[0] = spot_curve.iloc[0]

    epsilon = np.random.normal(scale=1, size=len(theta_t) - 1)

    for _t in range(1, len(theta_t) - 1):
        short_rate.loc[_t] = (
            short_rate.loc[_t - np.round(delta_t * 12)]
            + (theta_t.loc[_t] - k * short_rate.loc[_t - np.round(delta_t * 12)])
            * delta_t
            + sigma * np.sqrt(delta_t) * epsilon[_t - 1]
        )

    return short_rate


def hw(spot_curve:pd.Series, n:int=1, k:float=k, sigma:float=sigma, seed:int = 123) -> tuple:
    """Hull-White model

    Args:
        spot_curve (pd.Series): yield curve
        n (int, optional): number of short rate paths to generate. Defaults to 1.
        k (float): speed of mean reversion
        sigma (float): the standard deviation of rate of change in short rate
        seed (int, optional): seed. Defaults to 123.

    Returns:
        tuple: instantanous forward rates, theta(t) and short rate paths in order
    """
    instantaneous_fr = get_inst_fr(spot_curve, delta_t=1 / 12)
    theta_t = get_theta(instantaneous_fr=instantaneous_fr, k=k, sigma=sigma)

    np.random.seed(seed=seed)

    short_rate_path = pd.DataFrame()
    _p = 0
    while _p < n:
        short_rate_path = pd.merge(short_rate_path,
                                    get_short_rate(spot_curve, theta_t, k=k, sigma=sigma).rename(_p),
                                    left_index=True, 
                                    right_index=True,
                                    how="outer").sort_index(axis=1)
        _p += 1


    return (
            instantaneous_fr,
            theta_t,
            short_rate_path,
        )
    

def price_zero_coupon_bond(
    spot_curve: pd.Series,
    instantaneous_fr: pd.Series,
    short_rates: pd.DataFrame,
    maturity_date: int,
    t: int = 0,
    face_value: float = 100,
    k: float = k,
    sigma: float = sigma,
    # n_path: int = 1,
    # seed: int = 123,
    
) -> float:
    """this function calculate the value of a zero coupon bond at time t.

    Args:
        spot_curve (pd.Series): spot rates (treasury yields)
        maturity_date (int): the term when the bond matures (in month), T.
        t (int, optional): the time when the bond is valuated in month. Defaults to 0.
        face_value (float, optional): face value of the bond. Defaults to 100.
        k (float): for simplicity treated as a constant, defined in calibration section
        sigma (float): the standard deviation of rate of change in short rate, defined in calibration section
        n_path (int): the number of short rate paths from hw() used to price bonds
        seed (int): random seed. Defaults to 123.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        float: value of the zero-coupon bond at time t
    """

    # instantaneous_fr, _, short_rates = hw(spot_curve, n=n_path, seed=seed)

    T = maturity_date
    if T > 117:
        raise ValueError("Maturity exceeds short rate limits. ")
    if T < t:
        raise ValueError("Maturity must be positive.")

    if t == 0:
        P_t_T = face_value * np.exp(-spot_curve.loc[T] * T / 12)
    else:
        P_0_t = np.exp(-spot_curve.loc[t] * (t / 12))
        P_0_T = np.exp(-spot_curve.loc[T] * (T / 12))
        B_t_T = (1 - np.exp(-k * (T - t))) / k
        A_t_T = (P_0_T / P_0_t) * np.exp(
            B_t_T * instantaneous_fr.loc[t]
            - sigma**2 / (4 * k) * (1 - np.exp(-2 * k * (t / 12))) * B_t_T**2
        )

        # use the average of all the path
        P_t_T = face_value * A_t_T * np.exp(-B_t_T * short_rates.loc[t].mean())

    return P_t_T


def get_coupon_dates(
    t: int, 
    next_payment: int, 
    maturity_date: int, 
    coupon_interval: int
) -> list:
    """the function return the sequence of coupon payment date, [T1, T2, ... Tn] in month

    Args:
        t (int): valuation date
        next_payment_to_t (int): the number of months of the next coupon payment from t
        maturity_date (int): the term when the bond matures (in month), T.
        coupon_interval (int): 6 months or 12 months

    Returns:
        list: _description_
    """
    
    if next_payment < t:
        raise ValueError("The first coupon payment is earlier than the valuation date, i.e. next_payment < t.")

    c_payment_date = []

    # next_payment_to_t += t
    while next_payment <= maturity_date:
        c_payment_date.append(next_payment)
        next_payment += coupon_interval
    return c_payment_date

def get_fixed_rate_bond_price(
    spot_curve: pd.Series,
    instantaneous_fr: pd.Series,
    short_rates: pd.DataFrame,
    maturity_date: int,
    coupon_rate: float,
    next_payment: int,
    payment_frq: str = "semi",
    t: int = 0,
    face_value: float = 100,
    k: float = k,
    sigma: float = sigma,
    n_path: int = 1,
    seed: int = 123,
) -> float:
    """calculate the valuation of a fixed rate bond with simulated short rate

    Args:
        spot_curve (pd.Series): _description_
        time_to_maturity (int): in months
        maturity_date (int): the term when the bond matures (in month), T.
        coupon_rate (float): in percentage point
        next_payment (int): next coupon payment date in months
        payment_frq (str, optional): "semi" or "annual". Defaults to "semi".
        t (int, optional): in months. Defaults to 0.
        face_value (float, optional): Defaults to 100.
        k (float, optional): Defaults to k.
        sigma (float, optional): the standard deviation of rate of change in short rate. Defaults to sigma.
        n_path (int): the number of short rate paths from hw() used to price bonds
        seed (int): random seed. Defaults to 123.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        float: _description_
    """

    if payment_frq not in ["annual", "semi"]:
        raise ValueError(
            payment_frq
            + " is not a valid value for payment_frq; supported values are 'annual' and 'semi'."
        )

    coupon_interval = {"annual": 12, "semi": 6}[payment_frq]
    # check if the last coupon payment happens at the same time as the principle payment:
    if (
        maturity_date - next_payment
    ) % coupon_interval == 0 or maturity_date == next_payment:
        pass
    else:
        raise ValueError(
            "The last coupon payment does not happen at the same time as the principal payment."
        )

    # principal value at t
    principal_value_t = price_zero_coupon_bond(
        spot_curve,
        instantaneous_fr,
        short_rates,
        maturity_date=maturity_date,
        t=t,
        face_value=face_value,
        k=k,
        sigma=sigma,
        # n_path=n_path,
        # seed=seed,
    )

    if coupon_rate == 0:
        return principal_value_t

    # if the coupon is paid semi-annually, each payment will be half of the bond's annual coupon rate.
    coupon = (coupon_rate * face_value / 100) * (coupon_interval / 12)

    coupon_payment_date = get_coupon_dates(
        t, next_payment, maturity_date, coupon_interval
    )
    # print("Coupon payments will be made at times:")
    # print(*coupon_payment_date)

    total_coupon_value_t = 0

    for _c in coupon_payment_date:
        # consider each coupon payment as a zero-coupon bond
        this_coupon_v_t = price_zero_coupon_bond(
            spot_curve,
            instantaneous_fr,
            short_rates,
            maturity_date=_c,
            t=t,
            face_value=coupon,
            k=k,
            sigma=sigma,
            # n_path=n_path,
            # seed=seed,
        )
        total_coupon_value_t += this_coupon_v_t
        # print("the value of the coupon paid at time " + str(_c) + " is " + str(round(this_coupon_v_t, 6)) + " at time " + str(t))

    total_value = total_coupon_value_t + principal_value_t
    # print("The value of this fixed-rate bond is " + str(round(total_value, 4)) + " at time " + str(t))
    return total_value


