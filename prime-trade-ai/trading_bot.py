"""
Binance Futures Testnet Trading Bot
A simplified trading bot for placing market, limit, and stop-limit orders on Binance Futures Testnet.
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException


class BasicBot:
    """
    A basic trading bot for Binance Futures Testnet.
    Supports market, limit, and stop-limit orders for both buy and sell sides.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize the trading bot.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize logger
        self._setup_logger()
        
        # Initialize Binance client
        try:
            self.client = Client(api_key, api_secret, testnet=testnet)
            if testnet:
                self.client.API_URL = 'https://testnet.binancefuture.com'
            
            # Increase recv_window for timestamp tolerance
            self.client.timestamp_offset = 0
            
            self.logger.info("[OK] Bot initialized successfully")
            self.logger.info(f"[OK] Connected to {'Testnet' if testnet else 'Production'}")
            
            # Test connection
            self._test_connection()
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize bot: {str(e)}")
            raise
    
    def _setup_logger(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        import os
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # File handler for detailed logs (UTF-8 encoding for special characters)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/trading_bot_{timestamp}.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # Set UTF-8 encoding for console if possible
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass  # Fallback to default encoding
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _test_connection(self):
        """Test connection to Binance API."""
        try:
            self.logger.info("Testing API connection...")
            server_time = self.client.get_server_time()
            self.logger.info(f"[OK] Server time: {datetime.fromtimestamp(server_time['serverTime'] / 1000)}")
            
            # Sync local time with server time
            local_time = int(datetime.now().timestamp() * 1000)
            server_time_ms = server_time['serverTime']
            time_diff = server_time_ms - local_time
            if abs(time_diff) > 1000:  # If difference > 1 second
                self.client.timestamp_offset = time_diff
                self.logger.info(f"[OK] Adjusted timestamp offset: {time_diff}ms")
            
            # Get account info with increased recv_window
            account = self.client.futures_account(recvWindow=10000)
            self.logger.info(f"[OK] Account connected successfully")
            self.logger.debug(f"Account info: {account}")
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] API Error: {e.message} (Code: {e.code})")
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Connection test failed: {str(e)}")
            raise
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Dictionary containing account balance information
        """
        try:
            self.logger.info("Fetching account balance...")
            balance = self.client.futures_account_balance()
            self.logger.info("[OK] Balance retrieved successfully")
            self.logger.debug(f"Balance: {balance}")
            return balance
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] Failed to get balance: {e.message}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading rules and information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
        Returns:
            Dictionary containing symbol information
        """
        try:
            self.logger.info(f"Fetching symbol info for {symbol}...")
            exchange_info = self.client.futures_exchange_info()
            
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol.upper():
                    self.logger.info(f"[OK] Symbol info retrieved for {symbol}")
                    self.logger.debug(f"Symbol info: {s}")
                    return s
            
            raise ValueError(f"Symbol {symbol} not found")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get symbol info: {str(e)}")
            raise
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
        
        Returns:
            Dictionary containing order details
        """
        side = side.upper()
        symbol = symbol.upper()
        
        self.logger.info(f"Placing MARKET {side} order: {quantity} {symbol}")
        
        try:
            # Log request
            self.logger.debug(f"Request: symbol={symbol}, side={side}, type=MARKET, quantity={quantity}")
            
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Log response
            self.logger.info(f"[OK] Market order placed successfully")
            self.logger.info(f"  Order ID: {order['orderId']}")
            self.logger.info(f"  Status: {order['status']}")
            self.logger.info(f"  Executed Qty: {order['executedQty']}")
            self.logger.debug(f"Full response: {order}")
            
            return order
        except BinanceOrderException as e:
            self.logger.error(f"[ERROR] Order error: {e.message} (Code: {e.code})")
            raise
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] API error: {e.message} (Code: {e.code})")
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Unexpected error placing market order: {str(e)}")
            raise
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, 
                         price: float, time_in_force: str = 'GTC') -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
        
        Returns:
            Dictionary containing order details
        """
        side = side.upper()
        symbol = symbol.upper()
        time_in_force = time_in_force.upper()
        
        self.logger.info(f"Placing LIMIT {side} order: {quantity} {symbol} @ ${price}")
        
        try:
            # Log request
            self.logger.debug(f"Request: symbol={symbol}, side={side}, type=LIMIT, quantity={quantity}, price={price}, timeInForce={time_in_force}")
            
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                quantity=quantity,
                price=price,
                timeInForce=time_in_force
            )
            
            # Log response
            self.logger.info(f"[OK] Limit order placed successfully")
            self.logger.info(f"  Order ID: {order['orderId']}")
            self.logger.info(f"  Status: {order['status']}")
            self.logger.info(f"  Price: ${order['price']}")
            self.logger.debug(f"Full response: {order}")
            
            return order
        except BinanceOrderException as e:
            self.logger.error(f"[ERROR] Order error: {e.message} (Code: {e.code})")
            raise
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] API error: {e.message} (Code: {e.code})")
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Unexpected error placing limit order: {str(e)}")
            raise
    
    def place_stop_limit_order(self, symbol: str, side: str, quantity: float,
                              price: float, stop_price: float, 
                              time_in_force: str = 'GTC') -> Dict[str, Any]:
        """
        Place a stop-limit order (Bonus feature).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Limit price
            stop_price: Stop trigger price
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
        
        Returns:
            Dictionary containing order details
        """
        side = side.upper()
        symbol = symbol.upper()
        time_in_force = time_in_force.upper()
        
        self.logger.info(f"Placing STOP_LIMIT {side} order: {quantity} {symbol}")
        self.logger.info(f"  Stop Price: ${stop_price}, Limit Price: ${price}")
        
        try:
            # Log request
            self.logger.debug(f"Request: symbol={symbol}, side={side}, type=STOP, quantity={quantity}, price={price}, stopPrice={stop_price}, timeInForce={time_in_force}")
            
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP',
                quantity=quantity,
                price=price,
                stopPrice=stop_price,
                timeInForce=time_in_force
            )
            
            # Log response
            self.logger.info(f"[OK] Stop-limit order placed successfully")
            self.logger.info(f"  Order ID: {order['orderId']}")
            self.logger.info(f"  Status: {order['status']}")
            self.logger.info(f"  Stop Price: ${order['stopPrice']}")
            self.logger.info(f"  Limit Price: ${order['price']}")
            self.logger.debug(f"Full response: {order}")
            
            return order
        except BinanceOrderException as e:
            self.logger.error(f"[ERROR] Order error: {e.message} (Code: {e.code})")
            raise
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] API error: {e.message} (Code: {e.code})")
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Unexpected error placing stop-limit order: {str(e)}")
            raise
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID
        
        Returns:
            Dictionary containing order status
        """
        try:
            self.logger.info(f"Fetching order status for Order ID: {order_id}")
            order = self.client.futures_get_order(symbol=symbol.upper(), orderId=order_id)
            
            self.logger.info(f"[OK] Order status retrieved")
            self.logger.info(f"  Status: {order['status']}")
            self.logger.info(f"  Executed Qty: {order['executedQty']}")
            self.logger.debug(f"Full order details: {order}")
            
            return order
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] Failed to get order status: {e.message}")
            raise
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
        
        Returns:
            Dictionary containing cancellation details
        """
        try:
            self.logger.info(f"Cancelling order {order_id} for {symbol}")
            result = self.client.futures_cancel_order(symbol=symbol.upper(), orderId=order_id)
            
            self.logger.info(f"[OK] Order cancelled successfully")
            self.logger.debug(f"Cancellation details: {result}")
            
            return result
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] Failed to cancel order: {e.message}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter orders
        
        Returns:
            List of open orders
        """
        try:
            self.logger.info(f"Fetching open orders{' for ' + symbol if symbol else ''}...")
            
            if symbol:
                orders = self.client.futures_get_open_orders(symbol=symbol.upper())
            else:
                orders = self.client.futures_get_open_orders()
            
            self.logger.info(f"[OK] Found {len(orders)} open order(s)")
            self.logger.debug(f"Open orders: {orders}")
            
            return orders
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] Failed to get open orders: {e.message}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Current price as float
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol.upper())
            price = float(ticker['price'])
            self.logger.info(f"Current price for {symbol}: ${price}")
            return price
        except BinanceAPIException as e:
            self.logger.error(f"[ERROR] Failed to get current price: {e.message}")
            raise
