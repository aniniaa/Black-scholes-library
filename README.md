# Black-Scholes Options Pricing Library

A comprehensive Python library for options pricing using the Black-Scholes model and its extensions. This library aims to provide tools for pricing European and American options, calculating option Greeks, and analyzing option strategies.

## Features

- **Multiple Pricing Models**:
  - Black-Scholes model for European options
  - Barone-Adesi-Whaley approximation for American options
  - Bjerksund-Stensland approximation for American options
  - Binomial tree model for American options

- **Greeks Calculation**:
  - Delta (sensitivity to underlying price changes)
  - Gamma (second derivative of price with respect to underlying)
  - Theta (sensitivity to time decay, per day)
  - Vega (sensitivity to volatility changes)
  - Rho (sensitivity to interest rate changes)

- **Exotic Options**:
  - Binary (digital) options
  - Barrier options (up-and-in, up-and-out, down-and-in, down-and-out)
  - Asian options (using Monte Carlo simulation)

- **Implied Volatility Calculation**:
  - Bisection method
  - Newton-Raphson method

- **Option Strategies Analysis**:
  - Long Straddle
  - Bull Call Spread
  - Iron Condor

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/black-scholes-calculator.git

# Navigate to the directory
cd black-scholes-calculator

# Install dependencies
pip install numpy scipy matplotlib
```

## Usage Examples

### Basic Option Pricing

```python
from options_pricing_lib import OptionParameters, OptionType, OptionStyle, BlackScholesModel

# Create European call option parameters
call_params = OptionParameters(
    S=100,      # Current stock price
    K=100,      # Strike price
    r=0.05,     # Risk-free rate (5%)
    sigma=0.2,  # Volatility (20%)
    T=1,        # Time to maturity (1 year)
    q=0.02,     # Dividend yield (2%)
    option_type=OptionType.CALL,
    option_style=OptionStyle.EUROPEAN
)

# Calculate European call price
call_price = BlackScholesModel.price(call_params)
print(f"European Call Option Price: {call_price:.4f}")
```

### Calculating Greeks

```python
from options_pricing_lib import GreeksCalculator

# Calculate Delta
delta = GreeksCalculator.delta(call_params)
print(f"Delta: {delta:.4f}")

# Calculate all Greeks
gamma = GreeksCalculator.gamma(call_params)
theta = GreeksCalculator.theta(call_params)
vega = GreeksCalculator.vega(call_params)
rho = GreeksCalculator.rho(call_params)

print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f} (per day)")
print(f"Vega: {vega:.4f} (for 1% change in volatility)")
print(f"Rho: {rho:.4f} (for 1% change in interest rate)")
```

### American Options Pricing

```python
# Create American put option parameters
put_params = OptionParameters(
    S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.02,
    option_type=OptionType.PUT,
    option_style=OptionStyle.AMERICAN
)

# Calculate American put price using different methods
baw_price = BlackScholesModel.price(put_params, method="barone_adesi_whaley")
bs_price = BlackScholesModel.price(put_params, method="bjerksund_stensland")
binomial_price = BlackScholesModel.price(put_params, method="binomial")

print(f"American Put (Barone-Adesi-Whaley): {baw_price:.4f}")
print(f"American Put (Bjerksund-Stensland): {bs_price:.4f}")
print(f"American Put (Binomial Tree): {binomial_price:.4f}")
```

## Mathematical Foundation

This library implements the Black-Scholes-Merton model for option pricing, which is based on the following assumptions:
- The stock follows a geometric Brownian motion with constant drift and volatility
- No arbitrage opportunities
- Ability to buy and sell any amount of stock or option
- No transaction costs or taxes
- Risk-free interest rate remains constant
- All securities are perfectly divisible

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy.
- Barone-Adesi, G., & Whaley, R. (1987). Efficient Analytic Approximation of American Option Values.
- Bjerksund, P., & Stensland, G. (1993). Closed-form Approximation of American Options.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
