"""
Real-Time Options Pricing Extension
Adds live market data capabilities to the options pricing library
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

# Import from main library
from options_pricing_lib import (
    OptionType, OptionStyle, OptionParameters, 
    BlackScholesModel, GreeksCalculator
)

@dataclass
class LiveMarketData:
    """Container for live market data."""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    change_percent: float
    timestamp: datetime

class LiveDataProvider:
    """Provides live market data using Yahoo Finance."""
    
    def get_live_quote(self, symbol: str) -> LiveMarketData:
        """Get live market quote for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent data
            hist = ticker.history(period="2d", interval="1m")
            info = ticker.info
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
            
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest
            
            # Calculate change
            change_pct = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
            
            return LiveMarketData(
                symbol=symbol,
                price=float(latest['Close']),
                volume=int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                bid=info.get('bid', latest['Close']),
                ask=info.get('ask', latest['Close']),
                change_percent=change_pct,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def get_historical_volatility(self, symbol: str, period: str = "1y") -> float:
        """Calculate historical volatility."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            # Calculate log returns and annualized volatility
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            volatility = log_returns.std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.20  # Default 20%

class LiveOptionsCalculator:
    """Enhanced options calculator with live data."""
    
    def __init__(self):
        self.data_provider = LiveDataProvider()
    
    def get_live_option_price(self, 
                             symbol: str,
                             strike: float,
                             expiry_date: datetime,
                             option_type: OptionType,
                             risk_free_rate: float = 0.05) -> Dict[str, float]:
        """Calculate option price using live market data."""
        
        # Get live market data
        market_data = self.data_provider.get_live_quote(symbol)
        volatility = self.data_provider.get_historical_volatility(symbol)
        
        # Calculate time to expiry
        time_to_expiry = (expiry_date - datetime.now()).days / 365.0
        
        if time_to_expiry <= 0:
            #If option expired, return intrinsic value
            if option_type == OptionType.CALL:
                intrinsic = max(0, market_data.price - strike)
            else:
                intrinsic = max(0, strike - market_data.price)
            
            return {
                'option_price': intrinsic,
                'underlying_price': market_data.price,
                'status': 'expired',
                'intrinsic_value': intrinsic
            }
        
        # Create parameters for pricing
        params = OptionParameters(
            S=market_data.price,
            K=strike,
            r=risk_free_rate,
            sigma=volatility,
            T=time_to_expiry,
            q=0.0,  # Assume no dividends for simplicity
            option_type=option_type,
            option_style=OptionStyle.EUROPEAN
        )
        
        # Calculate option price and Greeks
        option_price = BlackScholesModel.price(params)
        
        # Calculate Greeks
        delta = GreeksCalculator.delta(params)
        gamma = GreeksCalculator.gamma(params)
        theta = GreeksCalculator.theta(params)
        vega = GreeksCalculator.vega(params)
        rho = GreeksCalculator.rho(params)
        
        # Calculate intrinsic and time value
        if option_type == OptionType.CALL:
            intrinsic_value = max(0, market_data.price - strike)
        else:
            intrinsic_value = max(0, strike - market_data.price)
        
        time_value = option_price - intrinsic_value
        
        return {
            'option_price': option_price,
            'intrinsic_value': intrinsic_value,
            'time_value': time_value,
            'underlying_price': market_data.price,
            'underlying_change_pct': market_data.change_percent,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'volume': market_data.volume,
            'implied_volatility': volatility,
            'time_to_expiry_days': time_to_expiry * 365,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'last_updated': market_data.timestamp,
            'status': 'active'
        }
    
    def get_options_chain_live(self, 
                              symbol: str,
                              expiry_date: datetime,
                              center_strike: Optional[float] = None,
                              strike_range: int = 10,
                              strike_step: float = 5.0) -> pd.DataFrame:
        """Generate live options chain around current price."""
        
        # Get current price to center strikes around
        if center_strike is None:
            market_data = self.data_provider.get_live_quote(symbol)
            center_strike = round(market_data.price / strike_step) * strike_step
        
        # Generate strike prices
        strikes = []
        for i in range(-strike_range, strike_range + 1):
            strikes.append(center_strike + (i * strike_step))
        
        # Calculate option prices for each strike
        results = []
        for strike in strikes:
            # Call option
            call_data = self.get_live_option_price(symbol, strike, expiry_date, OptionType.CALL)
            
            # Put option
            put_data = self.get_live_option_price(symbol, strike, expiry_date, OptionType.PUT)
            
            results.append({
                'strike': strike,
                'call_price': call_data['option_price'],
                'call_delta': call_data['delta'],
                'call_theta': call_data['theta'],
                'call_vega': call_data['vega'],
                'put_price': put_data['option_price'],
                'put_delta': put_data['delta'],
                'put_theta': put_data['theta'],
                'put_vega': put_data['vega'],
                'underlying_price': call_data['underlying_price']
            })
        
        df = pd.DataFrame(results)
        df['call_itm'] = df['strike'] < df['underlying_price']
        df['put_itm'] = df['strike'] > df['underlying_price']
        
        return df

class LivePortfolioMonitor:
    """Monitor a portfolio of options positions."""
    
    def __init__(self):
        self.calculator = LiveOptionsCalculator()
        self.positions = []
    
    def add_position(self, 
                    symbol: str,
                    strike: float,
                    expiry: datetime,
                    option_type: OptionType,
                    quantity: int,
                    entry_price: float):
        """Add a position to monitor."""
        
        position = {
            'id': len(self.positions),
            'symbol': symbol,
            'strike': strike,
            'expiry': expiry,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_date': datetime.now()
        }
        
        self.positions.append(position)
        print(f"✅ Added: {quantity} {symbol} {option_type.value} ${strike} @ ${entry_price}")
    
    def get_live_portfolio_status(self) -> Dict:
        """Get current portfolio status with live pricing."""
        
        total_value = 0
        total_pnl = 0
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        position_details = []
        
        for position in self.positions:
            try:
                # Get live option price
                live_data = self.calculator.get_live_option_price(
                    position['symbol'],
                    position['strike'],
                    position['expiry'],
                    position['option_type']
                )
                
                # Calculate position metrics
                current_price = live_data['option_price']
                position_value = current_price * position['quantity'] * 100  # Options multiplier
                entry_value = position['entry_price'] * position['quantity'] * 100
                position_pnl = position_value - entry_value
                
                # Greeks for this position
                pos_delta = live_data['delta'] * position['quantity']
                pos_gamma = live_data['gamma'] * position['quantity']
                pos_theta = live_data['theta'] * position['quantity']
                pos_vega = live_data['vega'] * position['quantity']
                
                # Aggregate totals
                total_value += position_value
                total_pnl += position_pnl
                total_delta += pos_delta
                total_gamma += pos_gamma
                total_theta += pos_theta
                total_vega += pos_vega
                
                # Store position details
                position_details.append({
                    'symbol': position['symbol'],
                    'type': position['option_type'].value,
                    'strike': position['strike'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'position_pnl': position_pnl,
                    'delta': pos_delta,
                    'gamma': pos_gamma,
                    'theta': pos_theta,
                    'vega': pos_vega,
                    'underlying_price': live_data['underlying_price'],
                    'days_to_expiry': live_data['time_to_expiry_days']
                })
                
            except Exception as e:
                print(f" Error updating {position['symbol']}: {e}")
        
        return {
            'timestamp': datetime.now(),
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'positions': position_details
        }
    
    def print_portfolio_report(self):
        """Print formatted portfolio report."""
        
        status = self.get_live_portfolio_status()
        
        print(f"LIVE PORTFOLIO REPORT - {status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Portfolio summary
        print(f"Portfolio Summary:")
        print(f"   Total Positions: {status['total_positions']}")
        print(f"   Portfolio Value: ${status['total_value']:,.2f})
        
        print(f"\n Portfolio Greeks:")
        print(f"   Delta: {status['total_delta']:.2f}")
        print(f"   Gamma: {status['total_gamma']:.4f}")
        print(f"   Theta: ${status['total_theta']:.2f}/day")
        print(f"   Vega: ${status['total_vega']:.2f}")
        
        # Individual positions
        print(f"\n Individual Positions:")
        print(f"{'Symbol':<6} {'Type':<4} {'Strike':<6} {'Qty':<4} {'Entry':<6} {'Current':<7} {'P&L':<10} {'Days':<4}")

        
        for pos in status['positions']:
            pnl_sign = "+" if pos['position_pnl'] >= 0 else ""
            print(f"{pos['symbol']:<6} "
                  f"{pos['type'][0]:<4} "
                  f"{pos['strike']:<6.0f} "
                  f"{pos['quantity']:<4} "
                  f"${pos['entry_price']:<5.2f} "
                  f"${pos['current_price']:<6.2f} "
                  f"{pnl_sign}${pos['position_pnl']:<9.2f} "
                  f"{pos['days_to_expiry']:<4.0f}")

# Demo functions
def demo_live_pricing():
    """Demonstrate live options pricing."""
    
    print(" LIVE OPTIONS PRICING DEMO")
    
    calculator = LiveOptionsCalculator()
    
    # Get live pricing for AAPL
    symbol = "AAPL"
    expiry = datetime.now() + timedelta(days=30)
    
    try:
        # Get current stock price first
        market_data = calculator.data_provider.get_live_quote(symbol)
        strike = round(market_data.price / 5) * 5  # Round to nearest $5
        
        print(f"Live data for {symbol}:")
        print(f"   Current Price: ${market_data.price:.2f}")
        print(f"   Change: {market_data.change_percent:+.2f}%")
        print(f"   Volume: {market_data.volume:,}")
        
        # Calculate call option
        call_data = calculator.get_live_option_price(symbol, strike, expiry, OptionType.CALL)
        
        print(f"\n Call Option (${strike} strike, {call_data['time_to_expiry_days']:.0f} days):")
        print(f"   Theoretical Price: ${call_data['option_price']:.2f}")
        print(f"   Intrinsic Value: ${call_data['intrinsic_value']:.2f}")
        print(f"   Time Value: ${call_data['time_value']:.2f}")
        print(f"   Delta: {call_data['delta']:.3f}")
        print(f"   Gamma: {call_data['gamma']:.4f}")
        print(f"   Theta: ${call_data['theta']:.2f}/day")
        print(f"   Vega: ${call_data['vega']:.2f}")
        print(f"   Implied Vol: {call_data['implied_volatility']:.1%}")
        
        # Calculate put option
        put_data = calculator.get_live_option_price(symbol, strike, expiry, OptionType.PUT)
        
        print(f"\n Put Option (${strike} strike):")
        print(f"   Theoretical Price: ${put_data['option_price']:.2f}")
        print(f"   Delta: {put_data['delta']:.3f}")
        
    except Exception as e:
        print(f" Error: {e}")

def demo_portfolio_monitoring():
    """Demonstrate portfolio monitoring."""
    
    print("\n LIVE PORTFOLIO MONITORING DEMO")
    
    portfolio = LivePortfolioMonitor()
    
    # Add some sample positions
    expiry = datetime.now() + timedelta(days=30)
    
    print("Adding sample positions...")
    portfolio.add_position("AAPL", 150, expiry, OptionType.CALL, 5, 6.50)
    portfolio.add_position("AAPL", 160, expiry, OptionType.CALL, -3, 3.20)  # Short
    portfolio.add_position("MSFT", 300, expiry, OptionType.PUT, 2, 12.75)
    
    # Show live portfolio status
    print("\nGenerating live portfolio report...")
    portfolio.print_portfolio_report()
