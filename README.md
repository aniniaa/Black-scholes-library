# Options Pricing Library

A comprehensive Python library for options pricing using the Black-Scholes model and its extensions, with real-time market data integration and portfolio management capabilities.

## Overview

This library provides institutional-grade options pricing models with real-time market data integration. It implements multiple pricing methodologies for European and American options, complete Greeks calculation, and supports exotic option types. The recent addition of real-time capabilities enables live portfolio tracking and market-based pricing.

## Key Features

### Pricing Models
- **Black-Scholes** model for European options
- **Barone-Adesi-Whaley** approximation for American options
- **Bjerksund-Stensland** approximation for American options
- **Cox-Ross-Rubinstein binomial tree** model

### Greeks Calculation
- Delta, Gamma, Theta, Vega, Rho
- Numerical differentiation with configurable precision
- Portfolio-level Greeks aggregation

### Exotic Options
- Binary (digital) options
- Barrier options (knock-in/knock-out variants)
- Asian options via Monte Carlo simulation

### Real-Time Integration
- Live market data from Yahoo Finance
- Real-time options pricing with current underlying prices
- Portfolio tracking with live P&L calculations
- Historical volatility estimation from market data
- Options chain generation with current pricing

### Advanced Analytics
- Implied volatility calculation (Bisection and Newton-Raphson methods)
- Option strategy analysis and visualization
- Risk metrics and portfolio optimization tools

## Installation

### From Source
```bash
git clone https://github.com/aniniaa/options-pricing-library.git
cd options-pricing-library
pip install -r requirements.txt
```

### Direct Installation
```bash
pip install git+https://github.com/aniniaa/options-pricing-library.git
```

## Usage Examples

### Basic Option Pricing

```python
from options_pricing_library import OptionParameters, OptionType, OptionStyle, BlackScholesModel

# Define option parameters
params = OptionParameters(
    S=100.0,    # Current stock price
    K=100.0,    # Strike price
    r=0.05,     # Risk-free rate
    sigma=0.2,  # Volatility
    T=1.0,      # Time to maturity (years)
    q=0.02,     # Dividend yield
    option_type=OptionType.CALL,
    option_style=OptionStyle.EUROPEAN
)

# Calculate option price
price = BlackScholesModel.price(params)
print(f"Option Price: ${price:.4f}")
```

### Greeks Calculation

```python
from options_pricing_library import GreeksCalculator

delta = GreeksCalculator.delta(params)
gamma = GreeksCalculator.gamma(params)
theta = GreeksCalculator.theta(params)
vega = GreeksCalculator.vega(params)
rho = GreeksCalculator.rho(params)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Vega: {vega:.4f}")
print(f"Rho: {rho:.4f}")
```

### Real-Time Option Pricing

```python
from options_pricing_library import LiveOptionsCalculator
from datetime import datetime, timedelta

calculator = LiveOptionsCalculator()
expiry = datetime.now() + timedelta(days=30)

# Get live option pricing
live_data = calculator.get_live_option_price("AAPL", 150, expiry, OptionType.CALL)

print(f"Underlying Price: ${live_data['underlying_price']:.2f}")
print(f"Option Price: ${live_data['option_price']:.2f}")
print(f"Delta: {live_data['delta']:.3f}")
print(f"Implied Volatility: {live_data['implied_volatility']:.1%}")
```

### Portfolio Management

```python
from options_pricing_library import LivePortfolioMonitor

portfolio = LivePortfolioMonitor()

# Add positions
portfolio.add_position("AAPL", 150, expiry, OptionType.CALL, 10, 6.50)
portfolio.add_position("MSFT", 300, expiry, OptionType.PUT, 5, 12.30)

# Generate portfolio report
status = portfolio.get_live_portfolio_status()
print(f"Portfolio Value: ${status['total_value']:,.2f}")
print(f"Total P&L: ${status['total_pnl']:,.2f}")
print(f"Portfolio Delta: {status['total_delta']:.2f}")
```

### American Options

```python
# American put option
american_params = OptionParameters(
    S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.02,
    option_type=OptionType.PUT,
    option_style=OptionStyle.AMERICAN
)

# Multiple pricing methods
baw_price = BlackScholesModel.price(american_params, method="barone_adesi_whaley")
bs_price = BlackScholesModel.price(american_params, method="bjerksund_stensland")
binomial_price = BlackScholesModel.price(american_params, method="binomial")
```

### Exotic Options

```python
from options_pricing_library import ExoticOptionsModel

# Binary option
binary_price = ExoticOptionsModel.binary_option_price(params, payoff=100)

# Barrier option
barrier_price = ExoticOptionsModel.barrier_option_price(
    params, barrier=110, barrier_type="up-and-out"
)

# Asian option
asian_price, std_error = ExoticOptionsModel.asian_option_monte_carlo(
    params, n_paths=10000, averaging_type="arithmetic"
)
```

### Implied Volatility

```python
market_price = 8.50
implied_vol = GreeksCalculator.implied_volatility(market_price, params)
print(f"Implied Volatility: {implied_vol:.1%}")
```

## Live Demo

Run the interactive demonstration:

```bash
python realtime_extension.py
```

Options include:
1. Live options pricing with real market data
2. Portfolio monitoring with real-time P&L tracking
3. Combined demonstration

## API Reference

### Core Classes

- **OptionParameters**: Container for option specification
- **BlackScholesModel**: Primary pricing engine
- **GreeksCalculator**: Sensitivity analysis
- **ExoticOptionsModel**: Specialized option types
- **LiveOptionsCalculator**: Real-time pricing integration
- **LivePortfolioMonitor**: Portfolio management and tracking

### Enumerations

- **OptionType**: CALL, PUT
- **OptionStyle**: EUROPEAN, AMERICAN

## Mathematical Foundation

The library implements the Black-Scholes-Merton framework with the following assumptions:

- Geometric Brownian motion for underlying asset prices
- Constant risk-free interest rate and volatility
- No transaction costs or taxes
- Continuous trading and perfect liquidity
- No dividends (or constant dividend yield)

## Performance Comparison

| Capability | This Library | Bloomberg Terminal |
|------------|-------------|-------------------|
| Real-time Data | Yahoo Finance | Professional feeds |
| Options Pricing | Multiple models | Industry standard |
| Greeks Calculation | Complete set | Advanced metrics |
| Customization | Full Python control | Limited scripting |
| Annual Cost | Free | $24,000+ |
| Transparency | Open source | Proprietary |
| Portfolio Tracking | Real-time | Professional tools |

## Dependencies

- numpy (≥1.21.0)
- scipy (≥1.7.0)
- pandas (≥1.3.0)
- matplotlib (≥3.4.0)
- yfinance (≥0.1.87)

## Use Cases

### Quantitative Research
- Model development and validation
- Strategy backtesting and optimization
- Academic research with reproducible results

### Trading Applications
- Real-time options pricing and Greeks monitoring
- Portfolio risk management
- Market making and arbitrage identification

### Education
- Options theory instruction with real market data
- Financial engineering coursework
- Professional development in quantitative finance

## Future updates to come:

- Additional pricing models and methodologies
- Enhanced data provider integration
- Advanced volatility modeling
- Performance optimizations
- Documentation improvements

Please submit pull requests with comprehensive tests and documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Barone-Adesi, G., & Whaley, R. E. (1987). Efficient Analytic Approximation of American Option Values. *Journal of Finance*, 42(2), 301-320.
- Bjerksund, P., & Stensland, G. (1993). Closed-Form Approximation of American Options. *Scandinavian Journal of Management*, 9, S87-S99.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229-263.
