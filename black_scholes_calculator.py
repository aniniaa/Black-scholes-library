import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Callable


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionParameters:
    """Parameters for option pricing."""

    S: float  # Current stock price
    K: float  # Strike price
    r: float  # Risk-free interest rate (annual)
    sigma: float  # Volatility (annual)
    T: float  # Time to maturity (in years)
    q: float = 0.0  # Dividend yield (annual)
    option_type: OptionType = OptionType.CALL
    option_style: OptionStyle = OptionStyle.EUROPEAN


class BlackScholesModel:
    """
    Black-Scholes model implementation for European options
    and approximation methods for American options.
    """

    @staticmethod
    def d1(params: OptionParameters) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        return (
            np.log(params.S / params.K)
            + (params.r - params.q + 0.5 * params.sigma**2) * params.T
        ) / (params.sigma * np.sqrt(params.T))

    @staticmethod
    def d2(params: OptionParameters) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return BlackScholesModel.d1(params) - params.sigma * np.sqrt(params.T)

    @staticmethod
    def european_option_price(params: OptionParameters) -> float:
        """
        Calculate European option price using Black-Scholes formula.

        Parameters:
            params: OptionParameters object containing option parameters

        Returns:
            float: Option price
        """
        d1 = BlackScholesModel.d1(params)
        d2 = BlackScholesModel.d2(params)

        if params.option_type == OptionType.CALL:
            return params.S * np.exp(-params.q * params.T) * norm.cdf(
                d1
            ) - params.K * np.exp(-params.r * params.T) * norm.cdf(d2)
        else:  # PUT
            return params.K * np.exp(-params.r * params.T) * norm.cdf(
                -d2
            ) - params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)

    @staticmethod
    def american_option_price(
        params: OptionParameters, method="barone_adesi_whaley"
    ) -> float:
        """
        Calculate American option price using various approximation methods.

        Parameters:
            params: OptionParameters object containing option parameters
            method: The approximation method to use
                    - "barone_adesi_whaley": Barone-Adesi and Whaley (1987) approximation
                    - "bjerksund_stensland": Bjerksund and Stensland (1993/2002) approximation
                    - "binomial": Cox-Ross-Rubinstein binomial tree model

        Returns:
            float: Option price
        """
        if params.option_style != OptionStyle.AMERICAN:
            raise ValueError("This method is only for American options")

        # For American options
        if method == "barone_adesi_whaley":
            return BlackScholesModel.barone_adesi_whaley(params)
        elif method == "bjerksund_stensland":
            return BlackScholesModel.bjerksund_stensland(params)
        elif method == "binomial":
            return BlackScholesModel.binomial_tree(params)
        else:
            raise ValueError(f"Method {method} not supported")

    @staticmethod
    def barone_adesi_whaley(params: OptionParameters) -> float:
        """
        Barone-Adesi and Whaley (1987) approximation for American options.

        This is an analytical approximation that works well for most cases.
        """
        # European option price as starting point
        european_price = BlackScholesModel.european_option_price(params)

        # For American calls with no dividends, value equals European calls
        if params.option_type == OptionType.CALL and params.q == 0:
            return european_price

        # For very short-dated options, use European price
        if params.T < 0.1:
            if params.option_type == OptionType.PUT:
                return max(params.K - params.S, european_price)
            else:
                return max(params.S - params.K, european_price)

        # Implementation of Barone-Adesi and Whaley (1987) approximation
        b = params.r - params.q
        M = 2 * params.r / (params.sigma**2)
        N = 2 * b / (params.sigma**2)

        # Calculate parameters
        if params.option_type == OptionType.CALL:
            q2 = (-N + np.sqrt(N**2 + 4 * M)) / 2

            def equation(S_star):
                d1_star = (
                    np.log(S_star / params.K) + (b + 0.5 * params.sigma**2) * params.T
                ) / (params.sigma * np.sqrt(params.T))
                A2 = (
                    S_star * (1 - np.exp(-params.q * params.T) * norm.cdf(d1_star)) / q2
                )
                return (
                    S_star
                    - params.K
                    - (
                        BlackScholesModel.european_option_price(
                            OptionParameters(
                                S=S_star,
                                K=params.K,
                                T=params.T,
                                r=params.r,
                                sigma=params.sigma,
                                q=params.q,
                                option_type=OptionType.CALL,
                                option_style=OptionStyle.EUROPEAN,
                            )
                        )
                        - (S_star - params.K)
                    )
                    - A2 * ((S_star / params.S) ** (-q2))
                )

            # Find the critical stock price S*
            try:
                S_star = brentq(equation, params.K, params.K * 5)
            except ValueError:
                # If we can't find S*, default to using S* = 2*K
                S_star = 2 * params.K

            d1_star = (
                np.log(S_star / params.K) + (b + 0.5 * params.sigma**2) * params.T
            ) / (params.sigma * np.sqrt(params.T))
            A2 = S_star * (1 - np.exp(-params.q * params.T) * norm.cdf(d1_star)) / q2

            if params.S < S_star:
                return european_price + A2 * ((params.S / S_star) ** q2)
            else:
                return params.S - params.K
        else:  # PUT
            q1 = (-N - np.sqrt(N**2 + 4 * M)) / 2

            def equation(S_star):
                d1_star = (
                    np.log(S_star / params.K) + (b + 0.5 * params.sigma**2) * params.T
                ) / (params.sigma * np.sqrt(params.T))
                A1 = -(S_star * np.exp(-params.q * params.T) * norm.cdf(-d1_star)) / q1
                return (
                    params.K
                    - S_star
                    - (
                        BlackScholesModel.european_option_price(
                            OptionParameters(
                                S=S_star,
                                K=params.K,
                                T=params.T,
                                r=params.r,
                                sigma=params.sigma,
                                q=params.q,
                                option_type=OptionType.PUT,
                                option_style=OptionStyle.EUROPEAN,
                            )
                        )
                        - (params.K - S_star)
                    )
                    - A1 * ((S_star / params.S) ** (-q1))
                )

            # Find the critical stock price S*
            try:
                S_star = brentq(equation, 0.001 * params.K, params.K)
            except ValueError:
                # If we can't find S*, default to using S* = 0.5*K
                S_star = 0.5 * params.K

            d1_star = (
                np.log(S_star / params.K) + (b + 0.5 * params.sigma**2) * params.T
            ) / (params.sigma * np.sqrt(params.T))
            A1 = -(S_star * np.exp(-params.q * params.T) * norm.cdf(-d1_star)) / q1

            if params.S > S_star:
                return european_price + A1 * ((params.S / S_star) ** q1)
            else:
                return params.K - params.S

    @staticmethod
    def bjerksund_stensland(params: OptionParameters) -> float:
        """
        Bjerksund and Stensland (1993/2002) approximation for American options.
        This method is particularly accurate for longer-term options.
        """
        # For American calls with no dividends, value equals European calls
        if params.option_type == OptionType.CALL and params.q == 0:
            return BlackScholesModel.european_option_price(params)

        # Implementation of the Bjerksund and Stensland approximation
        if params.option_type == OptionType.CALL:
            return BlackScholesModel._bjerksund_stensland_call(params)
        else:  # PUT using put-call transformation
            # Use the put-call transformation to value a put
            put_params = OptionParameters(
                S=params.K,
                K=params.S,
                r=params.q,
                q=params.r,
                sigma=params.sigma,
                T=params.T,
                option_type=OptionType.CALL,
                option_style=OptionStyle.AMERICAN,
            )
            return (
                BlackScholesModel._bjerksund_stensland_call(put_params)
                * params.K
                / params.S
            )

    @staticmethod
    def _bjerksund_stensland_call(params: OptionParameters) -> float:
        """Helper method for Bjerksund and Stensland call option pricing."""
        if params.S >= params.K:
            return max(
                params.S - params.K, BlackScholesModel.european_option_price(params)
            )

        b = params.r - params.q
        beta = (0.5 - b / (params.sigma**2)) + np.sqrt(
            (b / (params.sigma**2) - 0.5) ** 2 + 2 * params.r / (params.sigma**2)
        )

        # Calculate the boundary condition
        B_inf = beta / (beta - 1) * params.K
        B_zero = max(params.K, params.r / params.q * params.K)

        h1 = (
            -(b * params.T + 2 * params.sigma * np.sqrt(params.T))
            * params.K
            / (B_inf - params.K)
        )
        I = B_inf - (B_inf - params.K) * np.exp(h1)

        alpha = (I - params.K) * I ** (-beta)

        # The optimal exercise boundary
        if params.q <= 0:  # Never optimal to exercise before maturity
            H = I
        else:
            H = max(
                params.K,
                min(I, ((params.r / params.q) * params.K) if params.q > 0 else I),
            )

        if params.S >= H:
            return params.S - params.K

        # Calculate the option price using the approximation formula
        lambda_param = (-b + params.sigma**2 / 2.0) / (params.sigma**2)
        d1 = -(np.log(params.S / H) + (b + params.sigma**2 / 2) * params.T) / (
            params.sigma * np.sqrt(params.T)
        )
        d2 = d1 - params.sigma * np.sqrt(params.T)
        d3 = -(np.log(params.S / I) + (b + params.sigma**2 / 2) * params.T) / (
            params.sigma * np.sqrt(params.T)
        )
        d4 = d3 - params.sigma * np.sqrt(params.T)

        phi1 = BlackScholesModel._phi(
            params.S, params.T, beta, I, I, r=params.r, b=b, sigma=params.sigma
        )
        phi2 = BlackScholesModel._phi(
            params.S, params.T, 1, H, I, r=params.r, b=b, sigma=params.sigma
        )
        phi3 = BlackScholesModel._phi(
            params.S, params.T, 1, I, I, r=params.r, b=b, sigma=params.sigma
        )

        return (
            alpha * phi1
            - alpha * phi2
            + phi3
            - params.K * np.exp(-params.r * params.T) * norm.cdf(-d2)
            + params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)
        )

    @staticmethod
    def _phi(S, T, gamma, H, I, r, b, sigma):
        """Helper function for Bjerksund-Stensland formula."""
        lambda_param = (-b + gamma * sigma**2 / 2.0) / (sigma**2)

        d1 = -(np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        d2 = -(np.log(S / I) + (b + (gamma - 0.5) * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )

        return (
            np.exp(lambda_param * np.log(I / S))
            * S**gamma
            * (norm.cdf(d1) - (I / H) ** (2 * lambda_param) * norm.cdf(d2))
        )

    @staticmethod
    def binomial_tree(params: OptionParameters, steps: int = 100) -> float:
        """
        Cox-Ross-Rubinstein binomial tree model for American options.

        Parameters:
            params: OptionParameters object containing option parameters
            steps: Number of time steps in the binomial tree

        Returns:
            float: Option price
        """
        # Time step
        dt = params.T / steps

        # Calculate up and down factors
        u = np.exp(params.sigma * np.sqrt(dt))
        d = 1 / u

        # Risk-neutral probability
        p = (np.exp((params.r - params.q) * dt) - d) / (u - d)

        # Initialize asset prices at each node of the final time step
        prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            prices[i] = params.S * (u ** (steps - i)) * (d**i)

        # Initialize option values at each node of the final time step
        option_values = np.zeros(steps + 1)
        if params.option_type == OptionType.CALL:
            for i in range(steps + 1):
                option_values[i] = max(0, prices[i] - params.K)
        else:  # PUT
            for i in range(steps + 1):
                option_values[i] = max(0, params.K - prices[i])

        # Work backwards through the tree
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate the stock price at this node
                price = params.S * (u ** (step - i)) * (d**i)

                # Calculate the option value at this node (expected value)
                option_value = np.exp(-params.r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )

                # For American options, check for early exercise
                if params.option_type == OptionType.CALL:
                    option_values[i] = max(option_value, price - params.K)
                else:  # PUT
                    option_values[i] = max(option_value, params.K - price)

        return option_values[0]

    @staticmethod
    def price(params: OptionParameters, method: Optional[str] = None) -> float:
        """
        Calculate option price based on the option style and provided parameters.

        Parameters:
            params: OptionParameters object containing option parameters
            method: For American options, specify the approximation method
                   (ignored for European options)

        Returns:
            float: Option price
        """
        if params.option_style == OptionStyle.EUROPEAN:
            return BlackScholesModel.european_option_price(params)
        else:  # AMERICAN
            if method is None:
                # Default method selection based on option characteristics
                if params.T > 1:  # Longer-term options
                    method = "bjerksund_stensland"
                elif params.T < 0.1:  # Very short-term options
                    # For very short term, European is close to American
                    european = BlackScholesModel.european_option_price(params)
                    if params.option_type == OptionType.PUT:
                        return max(params.K - params.S, european)
                    else:
                        return max(params.S - params.K, european)
                else:  # Medium-term options
                    method = "barone_adesi_whaley"

            return BlackScholesModel.american_option_price(params, method=method)


class GreeksCalculator:
    """
    Class for calculating option Greeks (sensitivities).
    """

    @staticmethod
    def delta(params: OptionParameters, h: float = 0.01) -> float:
        """
        Calculate Delta (sensitivity to underlying price changes).

        Parameters:
            params: OptionParameters object containing option parameters
            h: Small change in stock price for numerical differentiation

        Returns:
            float: Delta value
        """
        params_up = OptionParameters(
            S=params.S + h,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        params_down = OptionParameters(
            S=params.S - h,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        price_up = BlackScholesModel.price(params_up)
        price_down = BlackScholesModel.price(params_down)

        return (price_up - price_down) / (2 * h)

    @staticmethod
    def gamma(params: OptionParameters, h: float = 0.01) -> float:
        """
        Calculate Gamma (second derivative of price with respect to underlying).

        Parameters:
            params: OptionParameters object containing option parameters
            h: Small change in stock price for numerical differentiation

        Returns:
            float: Gamma value
        """
        params_up = OptionParameters(
            S=params.S + h,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        params_down = OptionParameters(
            S=params.S - h,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        price_up = BlackScholesModel.price(params_up)
        price_center = BlackScholesModel.price(params)
        price_down = BlackScholesModel.price(params_down)

        return (price_up - 2 * price_center + price_down) / (h**2)

    @staticmethod
    def theta(params: OptionParameters, h: float = 0.01) -> float:
        """
        Calculate Theta (sensitivity to time decay).

        Parameters:
            params: OptionParameters object containing option parameters
            h: Small change in time for numerical differentiation

        Returns:
            float: Theta value (per day)
        """
        if params.T <= h:
            h = params.T / 2

        params_down = OptionParameters(
            S=params.S,
            K=params.K,
            r=params.r,
            sigma=params.sigma,
            T=params.T - h,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        price = BlackScholesModel.price(params)
        price_down = BlackScholesModel.price(params_down)

        # Convert to per day (252 trading days per year)
        return (price_down - price) / h * (1 / 252)

    @staticmethod
    def vega(params: OptionParameters, h: float = 0.01) -> float:
        """
        Calculate Vega (sensitivity to volatility changes).

        Parameters:
            params: OptionParameters object containing option parameters
            h: Small change in volatility for numerical differentiation

        Returns:
            float: Vega value (for a 1% change in volatility)
        """
        params_up = OptionParameters(
            S=params.S,
            K=params.K,
            r=params.r,
            sigma=params.sigma + h,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        price = BlackScholesModel.price(params)
        price_up = BlackScholesModel.price(params_up)

        # For a 1% change in volatility
        return (price_up - price) / h * 0.01

    @staticmethod
    def rho(params: OptionParameters, h: float = 0.01) -> float:
        """
        Calculate Rho (sensitivity to interest rate changes).

        Parameters:
            params: OptionParameters object containing option parameters
            h: Small change in interest rate for numerical differentiation

        Returns:
            float: Rho value (for a 1% change in interest rate)
        """
        params_up = OptionParameters(
            S=params.S,
            K=params.K,
            r=params.r + h,
            sigma=params.sigma,
            T=params.T,
            q=params.q,
            option_type=params.option_type,
            option_style=params.option_style,
        )

        price = BlackScholesModel.price(params)
        price_up = BlackScholesModel.price(params_up)

        # For a 1% change in interest rate
        return (price_up - price) / h * 0.01

    @staticmethod
    def implied_volatility(
        market_price: float,
        params: OptionParameters,
        method: str = "bisection",
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float:
        """
        Calculate implied volatility from market price.

        Parameters:
            market_price: Market price of the option
            params: OptionParameters object containing option parameters
            method: Method for root finding  "bisection" or "newton"
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations

        Returns:
            float: Implied volatility
        """
        if method == "bisection":
            return GreeksCalculator._implied_vol_bisection(
                market_price, params, tolerance, max_iterations
            )
        elif method == "newton":
            return GreeksCalculator._implied_vol_newton(
                market_price, params, tolerance, max_iterations
            )
        else:
            raise ValueError(f"Method {method} not supported")

    @staticmethod
    def _implied_vol_bisection(
        market_price: float,
        params: OptionParameters,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float:
        """Bisection method for implied volatility."""
        # Define bounds for volatility
        vol_low = 0.001
        vol_high = 5.0  # 500% volatility as upper bound

        # Initial guess in the middle
        vol_mid = (vol_low + vol_high) / 2

        for i in range(max_iterations):
            # Update sigma with current guess
            params_mid = OptionParameters(
                S=params.S,
                K=params.K,
                r=params.r,
                sigma=vol_mid,
                T=params.T,
                q=params.q,
                option_type=params.option_type,
                option_style=params.option_style,
            )

            # Calculate price with current volatility
            price_mid = BlackScholesModel.price(params_mid)

            # Check if we're close enough
            if abs(price_mid - market_price) < tolerance:
                return vol_mid

            # Update low and high bounds
            if price_mid > market_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

            # New midpoint
            vol_mid = (vol_low + vol_high) / 2

        # If we reach max iterations, return the best guess
        return vol_mid

    @staticmethod
    def _implied_vol_newton(
        market_price: float,
        params: OptionParameters,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float:
        """Newton-Raphson method for implied volatility."""
        # Initial guess for volatility
        vol = 0.3  # Start with 30% as initial guess

        for i in range(max_iterations):
            # Update sigma with current guess
            params_current = OptionParameters(
                S=params.S,
                K=params.K,
                r=params.r,
                sigma=vol,
                T=params.T,
                q=params.q,
                option_type=params.option_type,
                option_style=params.option_style,
            )

            # Calculate price with current volatility
            price = BlackScholesModel.price(params_current)

            # Calculate price difference
            diff = price - market_price

            # Check if we're close enough
            if abs(diff) < tolerance:
                return vol

            # Calculate vega (derivative of price with respect to volatility)
            vega = GreeksCalculator.vega(params_current, h=0.001) / 0.01

            # Newton-Raphson step: vol = vol - diff / vega
            # With damping to prevent large jumps
            vol = max(0.001, min(5.0, vol - 0.5 * diff / (vega + 1e-8)))

        # If we reach max iterations, return the best guess
        return vol


# Example usage
# Create a comprehensive visualization tool for options pricing
class OptionsPricingVisualizer:
    """
    Class for visualizing option prices and Greeks.
    """

    @staticmethod
    def plot_option_price_vs_underlying(
        params: OptionParameters, price_range: tuple = None, points: int = 100
    ):
        """
        Plot option price as a function of underlying price.

        Parameters:
            params: Base option parameters
            price_range: Tuple of (min_price, max_price) for the underlying
            points: Number of points to calculate for the plot
        """
        import matplotlib.pyplot as plt

        # If price range not specified, use a sensible default
        if price_range is None:
            price_range = (params.K * 0.5, params.K * 1.5)

        # Generate price points
        prices = np.linspace(price_range[0], price_range[1], points)

        # Calculate option prices
        eu_prices = []
        am_prices = []
        intrinsic_values = []

        for price in prices:
            # European option
            eu_params = OptionParameters(
                S=price,
                K=params.K,
                r=params.r,
                sigma=params.sigma,
                T=params.T,
                q=params.q,
                option_type=params.option_type,
                option_style=OptionStyle.EUROPEAN,
            )
            eu_prices.append(BlackScholesModel.price(eu_params))

            # American option
            am_params = OptionParameters(
                S=price,
                K=params.K,
                r=params.r,
                sigma=params.sigma,
                T=params.T,
                q=params.q,
                option_type=params.option_type,
                option_style=OptionStyle.AMERICAN,
            )
            am_prices.append(BlackScholesModel.price(am_params))

            # Intrinsic value
            if params.option_type == OptionType.CALL:
                intrinsic_values.append(max(0, price - params.K))
            else:  # PUT
                intrinsic_values.append(max(0, params.K - price))

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(prices, eu_prices, label="European")
        plt.plot(prices, am_prices, label="American")
        plt.plot(prices, intrinsic_values, "k--", label="Intrinsic Value")

        # Add the current price
        plt.axvline(
            x=params.S, color="r", linestyle=":", label=f"Current Price: {params.S}"
        )

        option_type_str = "Call" if params.option_type == OptionType.CALL else "Put"
        plt.title(
            f"{option_type_str} Option Price vs Underlying Price (K={params.K}, T={params.T}y)"
        )
        plt.xlabel("Underlying Price")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_greeks_vs_underlying(
        params: OptionParameters, price_range: tuple = None, points: int = 50
    ):
        """
        Plot option Greeks as a function of underlying price.

        Parameters:
            params: Base option parameters
            price_range: Tuple of (min_price, max_price) for the underlying
            points: Number of points to calculate for the plot
        """
        import matplotlib.pyplot as plt

        # If price range not specified, use a sensible default
        if price_range is None:
            price_range = (params.K * 0.7, params.K * 1.3)

        # Generate price points
        prices = np.linspace(price_range[0], price_range[1], points)

        # Calculate Greeks
        deltas = []
        gammas = []
        thetas = []
        vegas = []

        for price in prices:
            # Update parameters with current price
            current_params = OptionParameters(
                S=price,
                K=params.K,
                r=params.r,
                sigma=params.sigma,
                T=params.T,
                q=params.q,
                option_type=params.option_type,
                option_style=params.option_style,
            )

            # Calculate Greeks
            deltas.append(GreeksCalculator.delta(current_params))
            gammas.append(GreeksCalculator.gamma(current_params))
            thetas.append(GreeksCalculator.theta(current_params))
            vegas.append(GreeksCalculator.vega(current_params))

        # Plot the results
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        option_type_str = "Call" if params.option_type == OptionType.CALL else "Put"
        option_style_str = (
            "European" if params.option_style == OptionStyle.EUROPEAN else "American"
        )
        fig.suptitle(
            f"Option Greeks for {option_style_str} {option_type_str} (K={params.K}, T={params.T}y, σ={params.sigma})",
            fontsize=16,
        )

        # Delta
        axs[0, 0].plot(prices, deltas)
        axs[0, 0].set_title("Delta")
        axs[0, 0].axvline(x=params.K, color="r", linestyle=":", label="Strike Price")
        axs[0, 0].grid(True)

        # Gamma
        axs[0, 1].plot(prices, gammas)
        axs[0, 1].set_title("Gamma")
        axs[0, 1].axvline(x=params.K, color="r", linestyle=":", label="Strike Price")
        axs[0, 1].grid(True)

        # Theta
        axs[1, 0].plot(prices, thetas)
        axs[1, 0].set_title("Theta (per day)")
        axs[1, 0].axvline(x=params.K, color="r", linestyle=":", label="Strike Price")
        axs[1, 0].grid(True)

        # Vega
        axs[1, 1].plot(prices, vegas)
        axs[1, 1].set_title("Vega (for 1% change in volatility)")
        axs[1, 1].axvline(x=params.K, color="r", linestyle=":", label="Strike Price")
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_option_price_surface(
        params: OptionParameters,
        price_range: tuple = None,
        time_range: tuple = None,
        price_points: int = 20,
        time_points: int = 20,
    ):
        """
        Plot option price as a surface of underlying price and time to maturity.

        Parameters:
            params: Base option parameters
            price_range: Tuple of (min_price, max_price) for the underlying
            time_range: Tuple of (min_time, max_time) for time to maturity in years
            price_points: Number of price points
            time_points: Number of time points
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # If ranges not specified, use sensible defaults
        if price_range is None:
            price_range = (params.K * 0.6, params.K * 1.4)
        if time_range is None:
            time_range = (0.05, 2.0)  # From 0.05 years (about 2 weeks) to 2 years

        # Generate grid points
        prices = np.linspace(price_range[0], price_range[1], price_points)
        times = np.linspace(time_range[0], time_range[1], time_points)

        # Create meshgrid
        price_grid, time_grid = np.meshgrid(prices, times)

        # Calculate option prices
        option_prices = np.zeros((time_points, price_points))

        for i, t in enumerate(times):
            for j, price in enumerate(prices):
                # Update parameters
                current_params = OptionParameters(
                    S=price,
                    K=params.K,
                    r=params.r,
                    sigma=params.sigma,
                    T=t,
                    q=params.q,
                    option_type=params.option_type,
                    option_style=params.option_style,
                )

                # Calculate price
                option_prices[i, j] = BlackScholesModel.price(current_params)

        # Plot the results
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        option_type_str = "Call" if params.option_type == OptionType.CALL else "Put"
        option_style_str = (
            "European" if params.option_style == OptionStyle.EUROPEAN else "American"
        )

        surf = ax.plot_surface(
            price_grid, time_grid, option_prices, cmap="viridis", alpha=0.8
        )

        ax.set_xlabel("Underlying Price")
        ax.set_ylabel("Time to Maturity (years)")
        ax.set_zlabel("Option Price")

        ax.set_title(
            f"{option_style_str} {option_type_str} Option Price Surface (K={params.K}, σ={params.sigma})"
        )

        # Add a colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.show()


# Extension to price Exotic Options
class ExoticOptionsModel:
    """
    Class for pricing exotic options.
    """

    @staticmethod
    def binary_option_price(params: OptionParameters, payoff: float = 1.0) -> float:
        """
        Price a binary (digital) option.

        Parameters:
            params: Option parameters
            payoff: The fixed payoff amount (default: 1.0)

        Returns:
            float: Binary option price
        """
        # Only European style is supported for binary options
        if params.option_style != OptionStyle.EUROPEAN:
            raise ValueError("Binary options only support European style")

        d2 = BlackScholesModel.d2(params)

        if params.option_type == OptionType.CALL:
            # Binary call pays if S > K at expiry
            return payoff * np.exp(-params.r * params.T) * norm.cdf(d2)
        else:
            # Binary put pays if S < K at expiry
            return payoff * np.exp(-params.r * params.T) * norm.cdf(-d2)

    @staticmethod
    def barrier_option_price(
        params: OptionParameters, barrier: float, barrier_type: str, rebate: float = 0.0
    ) -> float:
        """
        Price a barrier option using analytical formulas.

        Parameters:
            params: Option parameters
            barrier: Barrier level
            barrier_type: One of "up-and-in", "up-and-out", "down-and-in", "down-and-out"
            rebate: Rebate amount paid if barrier is hit (or not hit for out options)

        Returns:
            float: Barrier option price
        """
        # Only European style is supported for analytical barrier pricing
        if params.option_style != OptionStyle.EUROPEAN:
            raise ValueError("Analytical barrier pricing only supports European style")

        S = params.S
        K = params.K
        r = params.r
        q = params.q
        T = params.T
        sigma = params.sigma
        H = barrier  # Barrier level

        # Calculate standard parameters
        d1 = BlackScholesModel.d1(params)
        d2 = BlackScholesModel.d2(params)

        # Calculate additional parameters for barrier options
        mu = (r - q) / (sigma**2) - 0.5
        lambda_param = np.sqrt(mu**2 + 2 * r / (sigma**2))

        # Calculate powers
        x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y1 = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

        # Helper function for the barrier formulas
        def phi(x):
            return norm.cdf(x)

        # Standard Black-Scholes price (for reference)
        if params.option_type == OptionType.CALL:
            vanilla = S * np.exp(-q * T) * phi(d1) - K * np.exp(-r * T) * phi(d2)
        else:  # PUT
            vanilla = K * np.exp(-r * T) * phi(-d2) - S * np.exp(-q * T) * phi(-d1)

        # Different barrier option types
        if barrier_type == "down-and-out":
            if params.option_type == OptionType.CALL:
                if S <= H:
                    return rebate * np.exp(-r * T)  # Knocked out immediately

                # Down-and-out call
                term1 = S * np.exp(-q * T) * phi(d1) - K * np.exp(-r * T) * phi(d2)
                term2 = (
                    S
                    * np.exp(-q * T)
                    * (H / S) ** (2 * mu + 2)
                    * phi(-x1 + 2 * mu * sigma * np.sqrt(T))
                )
                term3 = (
                    K
                    * np.exp(-r * T)
                    * (H / S) ** (2 * mu)
                    * phi(-y1 + 2 * mu * sigma * np.sqrt(T))
                )

                return term1 - term2 + term3 + rebate * np.exp(-r * T) * phi(-y1)
            else:  # PUT
                if S <= H:
                    return rebate * np.exp(-r * T)  # Knocked out immediately

                # Down-and-out put
                return (
                    vanilla
                    - (S * np.exp(-q * T) * phi(-d1) - K * np.exp(-r * T) * phi(-d2))
                    * (H / S) ** (2 * mu + 2)
                    + rebate * np.exp(-r * T) * phi(-y1)
                )

        elif barrier_type == "up-and-out":
            if params.option_type == OptionType.CALL:
                if S >= H:
                    return rebate * np.exp(-r * T)  # Knocked out immediately

                # Up-and-out call
                return (
                    vanilla
                    - (S * np.exp(-q * T) * phi(d1) - K * np.exp(-r * T) * phi(d2))
                    * (S / H) ** (2 * lambda_param)
                    + rebate * np.exp(-r * T) * phi(x1)
                )
            else:  # PUT
                if S >= H:
                    return rebate * np.exp(-r * T)  # Knocked out immediately

                # Up-and-out put
                term1 = K * np.exp(-r * T) * phi(-d2) - S * np.exp(-q * T) * phi(-d1)
                term2 = (
                    S
                    * np.exp(-q * T)
                    * (S / H) ** (2 * mu + 2)
                    * phi(x1 + 2 * mu * sigma * np.sqrt(T))
                )
                term3 = (
                    K
                    * np.exp(-r * T)
                    * (S / H) ** (2 * mu)
                    * phi(y1 + 2 * mu * sigma * np.sqrt(T))
                )

                return term1 - term2 + term3 + rebate * np.exp(-r * T) * phi(x1)

        elif barrier_type == "down-and-in":
            if params.option_type == OptionType.CALL:
                if S <= H:
                    return vanilla  # Already knocked in

                # Down-and-in call = vanilla - down-and-out call
                term1 = (
                    S
                    * np.exp(-q * T)
                    * (H / S) ** (2 * mu + 2)
                    * phi(-x1 + 2 * mu * sigma * np.sqrt(T))
                )
                term2 = (
                    K
                    * np.exp(-r * T)
                    * (H / S) ** (2 * mu)
                    * phi(-y1 + 2 * mu * sigma * np.sqrt(T))
                )

                return term1 - term2 - rebate * np.exp(-r * T) * phi(-y1)
            else:  # PUT
                if S <= H:
                    return vanilla  # Already knocked in

                # Down-and-in put = vanilla - down-and-out put
                return (
                    S * np.exp(-q * T) * phi(-d1) - K * np.exp(-r * T) * phi(-d2)
                ) * (H / S) ** (2 * mu + 2) - rebate * np.exp(-r * T) * phi(-y1)

        elif barrier_type == "up-and-in":
            if params.option_type == OptionType.CALL:
                if S >= H:
                    return vanilla  # Already knocked in

                # Up-and-in call = vanilla - up-and-out call
                return (S * np.exp(-q * T) * phi(d1) - K * np.exp(-r * T) * phi(d2)) * (
                    S / H
                ) ** (2 * lambda_param) - rebate * np.exp(-r * T) * phi(x1)
            else:  # PUT
                if S >= H:
                    return vanilla  # Already knocked in

                # Up-and-in put = vanilla - up-and-out put
                term1 = (
                    S
                    * np.exp(-q * T)
                    * (S / H) ** (2 * mu + 2)
                    * phi(x1 + 2 * mu * sigma * np.sqrt(T))
                )
                term2 = (
                    K
                    * np.exp(-r * T)
                    * (S / H) ** (2 * mu)
                    * phi(y1 + 2 * mu * sigma * np.sqrt(T))
                )

                return term1 - term2 - rebate * np.exp(-r * T) * phi(x1)
        else:
            raise ValueError(f"Barrier type {barrier_type} not supported")

    @staticmethod
    def asian_option_monte_carlo(
        params: OptionParameters,
        n_paths: int = 10000,
        n_steps: int = 252,
        averaging_type: str = "arithmetic",
        seed: int = None,
    ) -> float:
        """
        Price an Asian option using Monte Carlo simulation.

        Parameters:
            params: Option parameters
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path (typically trading days in a year)
            averaging_type: "arithmetic" or "geometric" averaging
            seed: Random seed for reproducibility

        Returns:
            float: Asian option price with standard error estimate
        """
        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Parameters
        S0 = params.S
        K = params.K
        r = params.r
        q = params.q
        sigma = params.sigma
        T = params.T

        # Time step
        dt = T / n_steps

        # Generate paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Random normal variates for simulation
        Z = np.random.normal(size=(n_paths, n_steps))

        # Simulate paths
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        # Calculate average prices (excluding initial price)
        if averaging_type == "arithmetic":
            avg_prices = np.mean(paths[:, 1:], axis=1)
        elif averaging_type == "geometric":
            avg_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        else:
            raise ValueError(f"Averaging type {averaging_type} not supported")

        # Calculate payoffs
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(0, avg_prices - K)
        else:  # PUT
            payoffs = np.maximum(0, K - avg_prices)

        # Discount payoffs to present value
        discounted_payoffs = np.exp(-r * T) * payoffs

        # Price and standard error
        price = np.mean(discounted_payoffs)
        se = np.std(discounted_payoffs) / np.sqrt(n_paths)

        return price, se


def main():
    """
    Main demonstration function for the Black-Scholes Options Pricing Library.
    """
    print("Black-Scholes Options Pricing Library - Demonstration")

    # Basic European and American option pricing
    print("1. Basic Option Pricing Examples")

    # Create option parameters for a European call option
    european_call_params = OptionParameters(
        S=100,  # Current stock price
        K=100,  # Strike price
        r=0.05,  # Risk-free rate (5%)
        sigma=0.2,  # Volatility (20%)
        T=1,  # Time to maturity (1 year)
        q=0.02,  # Dividend yield (2%)
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )

    # Calculate European call option price
    european_call_price = BlackScholesModel.price(european_call_params)
    print(f"European Call Option Price: {european_call_price:.4f}")

    # Create option parameters for an American put option
    american_put_params = OptionParameters(
        S=100,  # Current stock price
        K=100,  # Strike price
        r=0.05,  # Risk-free rate (5%)
        sigma=0.2,  # Volatility (20%)
        T=1,  # Time to maturity (1 year)
        q=0.02,  # Dividend yield (2%)
        option_type=OptionType.PUT,
        option_style=OptionStyle.AMERICAN,
    )

    # Calculate American put option price using different methods
    baw_price = BlackScholesModel.price(
        american_put_params, method="barone_adesi_whaley"
    )
    bs_price = BlackScholesModel.price(
        american_put_params, method="bjerksund_stensland"
    )
    binomial_price = BlackScholesModel.price(american_put_params, method="binomial")

    print(f"American Put Option Price (Barone-Adesi-Whaley): {baw_price:.4f}")
    print(f"American Put Option Price (Bjerksund-Stensland): {bs_price:.4f}")
    print(f"American Put Option Price (Binomial Tree): {binomial_price:.4f}")

    # Calculate Greeks
    print("\n2. Option Greeks Calculation")
    delta = GreeksCalculator.delta(european_call_params)
    gamma = GreeksCalculator.gamma(european_call_params)
    theta = GreeksCalculator.theta(european_call_params)
    vega = GreeksCalculator.vega(european_call_params)
    rho = GreeksCalculator.rho(european_call_params)

    print(f"Greeks for European Call Option:")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f} (per day)")
    print(f"Vega: {vega:.4f} (for 1% change in volatility)")
    print(f"Rho: {rho:.4f} (for 1% change in interest rate)")

    # Calculate implied volatility
    market_price = 10.0  # Assume we observed this price in the market
    implied_vol = GreeksCalculator.implied_volatility(
        market_price, european_call_params
    )
    print(
        f"\nImplied Volatility: {implied_vol:.4f} (from market price: {market_price:.2f})"
    )

    # Volatility effects
    print("\n3. Effect of Volatility on Option Prices")
    volatilities = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("Volatility | European Call | European Put | American Call | American Put")
    print("---------+---------------+-------------+---------------+------------")

    for vol in volatilities:
        # Update parameters with current volatility
        eu_call_params = OptionParameters(
            S=100,
            K=100,
            r=0.05,
            sigma=vol,
            T=1,
            q=0.02,
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
        )
        eu_put_params = OptionParameters(
            S=100,
            K=100,
            r=0.05,
            sigma=vol,
            T=1,
            q=0.02,
            option_type=OptionType.PUT,
            option_style=OptionStyle.EUROPEAN,
        )
        am_call_params = OptionParameters(
            S=100,
            K=100,
            r=0.05,
            sigma=vol,
            T=1,
            q=0.02,
            option_type=OptionType.CALL,
            option_style=OptionStyle.AMERICAN,
        )
        am_put_params = OptionParameters(
            S=100,
            K=100,
            r=0.05,
            sigma=vol,
            T=1,
            q=0.02,
            option_type=OptionType.PUT,
            option_style=OptionStyle.AMERICAN,
        )

        # Calculate prices
        eu_call = BlackScholesModel.price(eu_call_params)
        eu_put = BlackScholesModel.price(eu_put_params)
        am_call = BlackScholesModel.price(am_call_params)
        am_put = BlackScholesModel.price(am_put_params)

        print(
            f"{vol:.1f}      | ${eu_call:.2f}        | ${eu_put:.2f}      | ${am_call:.2f}        | ${am_put:.2f}"
        )

    # Put-call parity
    print("\n4. Put-Call Parity Verification")
    s = 100
    k = 100
    r = 0.05
    q = 0.02
    t = 1
    sigma = 0.2

    # Calculate European call and put prices
    eu_call = BlackScholesModel.price(
        OptionParameters(
            S=s,
            K=k,
            r=r,
            sigma=sigma,
            T=t,
            q=q,
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
        )
    )

    eu_put = BlackScholesModel.price(
        OptionParameters(
            S=s,
            K=k,
            r=r,
            sigma=sigma,
            T=t,
            q=q,
            option_type=OptionType.PUT,
            option_style=OptionStyle.EUROPEAN,
        )
    )

    # Check put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
    lhs = eu_call - eu_put
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)

    print(f"Call Price: ${eu_call:.2f}")
    print(f"Put Price: ${eu_put:.2f}")
    print(f"LHS (C - P): ${lhs:.2f}")
    print(f"RHS (S*e^(-q*T) - K*e^(-r*T)): ${rhs:.2f}")
    print(f"Difference: ${abs(lhs - rhs):.6f}")

    if abs(lhs - rhs) < 1e-10:
        print("Put-Call Parity is verified! ✓")
    else:
        print("Put-Call Parity check failed!")

    # Exotic options
    print("\n5. Exotic Options Pricing")

    # Binary option
    binary_params = OptionParameters(
        S=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1,
        q=0.02,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )
    binary_price = ExoticOptionsModel.binary_option_price(binary_params, payoff=100)
    print(f"Binary Call Option Price (payoff=$100): ${binary_price:.2f}")

    # Barrier options
    barrier_params = OptionParameters(
        S=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1,
        q=0.02,
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
    )

    # Down-and-out put
    do_price = ExoticOptionsModel.barrier_option_price(
        barrier_params, barrier=80, barrier_type="down-and-out"
    )
    print(f"Down-and-out Put Barrier Option Price (barrier=80): ${do_price:.2f}")

    # Up-and-in call
    ui_params = OptionParameters(
        S=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1,
        q=0.02,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )
    ui_price = ExoticOptionsModel.barrier_option_price(
        ui_params, barrier=120, barrier_type="up-and-in"
    )
    print(f"Up-and-in Call Barrier Option Price (barrier=120): ${ui_price:.2f}")

    # Asian option using Monte Carlo
    asian_params = OptionParameters(
        S=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1,
        q=0.02,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )

    # Arithmetic average
    asian_price, se = ExoticOptionsModel.asian_option_monte_carlo(
        asian_params, n_paths=10000, averaging_type="arithmetic", seed=42
    )
    print(f"Asian Call Option Price (arithmetic avg): ${asian_price:.2f} ± ${se:.2f}")

    # Geometric average
    asian_geo_price, geo_se = ExoticOptionsModel.asian_option_monte_carlo(
        asian_params, n_paths=10000, averaging_type="geometric", seed=42
    )
    print(
        f"Asian Call Option Price (geometric avg): ${asian_geo_price:.2f} ± ${geo_se:.2f}"
    )

    # Option strategies
    print("\n6. Option Strategy Analysis")
    current_price = 100
    strike = 100
    vol = 0.2
    r = 0.05
    T = 0.5  # 6 months

    # Long Straddle (Long Call + Long Put at the same strike)
    call_params = OptionParameters(
        S=current_price,
        K=strike,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )
    put_params = OptionParameters(
        S=current_price,
        K=strike,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
    )

    call_price = BlackScholesModel.price(call_params)
    put_price = BlackScholesModel.price(put_params)

    # Define the strategy
    long_straddle = [
        (call_params, 1, call_price),  # Long 1 call, paying call_price premium
        (put_params, 1, put_price),  # Long 1 put, paying put_price premium
    ]

    # Analyze the strategy
    print("\nStrategy: Long Straddle")
    total_premium = call_price + put_price
    print(f"Call Premium: ${call_price:.2f}")
    print(f"Put Premium: ${put_price:.2f}")
    print(f"Total Premium Paid: ${total_premium:.2f}")
    print(
        f"Break-even points: ${strike - total_premium:.2f} and ${strike + total_premium:.2f}"
    )
    print(f"Max Loss: ${total_premium:.2f} (if stock price = ${strike} at expiration)")
    print(f"Max Profit: Unlimited (the further the stock moves from ${strike})")

    # Bull Call Spread
    print("\nStrategy: Bull Call Spread")
    lower_strike = 95
    higher_strike = 105

    # Calculate option prices
    lower_call_params = OptionParameters(
        S=current_price,
        K=lower_strike,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )
    higher_call_params = OptionParameters(
        S=current_price,
        K=higher_strike,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )

    lower_call_price = BlackScholesModel.price(lower_call_params)
    higher_call_price = BlackScholesModel.price(higher_call_params)

    net_debit = lower_call_price - higher_call_price
    max_profit = higher_strike - lower_strike - net_debit

    print(f"Long Call @ ${lower_strike} Strike: ${lower_call_price:.2f}")
    print(f"Short Call @ ${higher_strike} Strike: ${higher_call_price:.2f}")
    print(f"Net Debit: ${net_debit:.2f}")
    print(
        f"Max Profit: ${max_profit:.2f} (if stock price >= ${higher_strike} at expiration)"
    )
    print(
        f"Max Loss: ${net_debit:.2f} (if stock price <= ${lower_strike} at expiration)"
    )
    print(f"Break-even: ${lower_strike + net_debit:.2f}")

    # Iron Condor
    print("\nStrategy: Iron Condor")
    put_spread_lower = 85
    put_spread_upper = 90
    call_spread_lower = 110
    call_spread_upper = 115

    # Calculate option prices
    put1_params = OptionParameters(
        S=current_price,
        K=put_spread_lower,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
    )
    put2_params = OptionParameters(
        S=current_price,
        K=put_spread_upper,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
    )
    call1_params = OptionParameters(
        S=current_price,
        K=call_spread_lower,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )
    call2_params = OptionParameters(
        S=current_price,
        K=call_spread_upper,
        r=r,
        sigma=vol,
        T=T,
        q=0.0,
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
    )

    put1_price = BlackScholesModel.price(put1_params)
    put2_price = BlackScholesModel.price(put2_params)
    call1_price = BlackScholesModel.price(call1_params)
    call2_price = BlackScholesModel.price(call2_params)

    put_spread_credit = put2_price - put1_price
    call_spread_credit = call1_price - call2_price
    total_credit = put_spread_credit + call_spread_credit
    max_risk = (put_spread_upper - put_spread_lower) - total_credit

    print(f"Short Put @ ${put_spread_lower} Strike: ${put1_price:.2f}")
    print(f"Long Put @ ${put_spread_upper} Strike: ${put2_price:.2f}")
    print(f"Short Call @ ${call_spread_lower} Strike: ${call1_price:.2f}")
    print(f"Long Call @ ${call_spread_upper} Strike: ${call2_price:.2f}")
    print(f"Total Net Credit: ${total_credit:.2f}")
    print(
        f"Max Profit: ${total_credit:.2f} (if ${put_spread_upper} <= stock <= ${call_spread_lower} at expiration)"
    )
    print(
        f"Max Loss: ${max_risk:.2f} (if stock <= ${put_spread_lower} or stock >= ${call_spread_upper})"
    )
    print(f"Lower Break-even: ${put_spread_upper - total_credit:.2f}")
    print(f"Upper Break-even: ${call_spread_lower + total_credit:.2f}")

    print("\nThank you for using the Black-Scholes Options Pricing Library!")


if __name__ == "__main__":
    main()
