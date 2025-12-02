"""
Command-Line Interface for Binance Futures Trading Bot
Provides an interactive CLI for placing and managing orders.
"""

import sys
import os
from typing import Optional
from colorama import init, Fore, Style
from dotenv import load_dotenv
from trading_bot import BasicBot
from binance.exceptions import BinanceAPIException


# Initialize colorama for Windows support
init(autoreset=True)


class TradingBotCLI:
    """Interactive command-line interface for the trading bot."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.bot: Optional[BasicBot] = None
        self.running = True
    
    def print_header(self):
        """Print the CLI header."""
        print(f"\n{Fore.CYAN}{'=' * 70}")
        print(f"{Fore.CYAN}{'Binance Futures Testnet Trading Bot':^70}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")
    
    def print_menu(self):
        """Print the main menu."""
        print(f"\n{Fore.YELLOW}{'─' * 70}")
        print(f"{Fore.GREEN}Main Menu:")
        print(f"{Fore.WHITE}  1. Place Market Order")
        print(f"  2. Place Limit Order")
        print(f"  3. Place Stop-Limit Order (Bonus)")
        print(f"  4. Check Order Status")
        print(f"  5. View Open Orders")
        print(f"  6. Cancel Order")
        print(f"  7. Check Account Balance")
        print(f"  8. Get Current Price")
        print(f"  9. View Symbol Info")
        print(f"  0. Exit")
        print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")
    
    def get_input(self, prompt: str, input_type=str, allow_empty: bool = False):
        """
        Get validated user input.
        
        Args:
            prompt: Prompt message
            input_type: Expected type (str, float, int)
            allow_empty: Whether to allow empty input
        
        Returns:
            Validated input value
        """
        while True:
            try:
                value = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
                
                if not value and allow_empty:
                    return None
                
                if not value:
                    print(f"{Fore.RED}[ERROR] Input cannot be empty{Style.RESET_ALL}")
                    continue
                
                if input_type == str:
                    return value
                elif input_type == float:
                    return float(value)
                elif input_type == int:
                    return int(value)
            except ValueError:
                print(f"{Fore.RED}[ERROR] Invalid input. Please enter a valid {input_type.__name__}{Style.RESET_ALL}")
    
    def get_side(self) -> str:
        """Get order side (BUY/SELL) from user."""
        while True:
            side = self.get_input("Enter order side (BUY/SELL): ").upper()
            if side in ['BUY', 'SELL']:
                return side
            print(f"{Fore.RED}[ERROR] Invalid side. Please enter BUY or SELL{Style.RESET_ALL}")
    
    def initialize_bot(self):
        """Initialize the trading bot with API credentials."""
        print(f"\n{Fore.YELLOW}Initializing Trading Bot...{Style.RESET_ALL}\n")
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # If not in .env, ask user
        if not api_key or api_key == 'your_api_key_here':
            print(f"{Fore.YELLOW}API credentials not found in .env file{Style.RESET_ALL}")
            api_key = self.get_input("Enter your Binance API Key: ")
            api_secret = self.get_input("Enter your Binance API Secret: ")
        
        try:
            self.bot = BasicBot(api_key, api_secret, testnet=True)
            print(f"\n{Fore.GREEN}[OK] Bot initialized successfully!{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Failed to initialize bot: {str(e)}{Style.RESET_ALL}")
            return False
    
    def place_market_order_menu(self):
        """Handle market order placement."""
        print(f"\n{Fore.YELLOW}── Place Market Order ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        side = self.get_side()
        quantity = self.get_input("Enter quantity: ", float)
        
        confirm = self.get_input(f"\nConfirm MARKET {side} order: {quantity} {symbol}? (yes/no): ").lower()
        
        if confirm == 'yes':
            try:
                order = self.bot.place_market_order(symbol, side, quantity)
                print(f"\n{Fore.GREEN}[OK] Order placed successfully!{Style.RESET_ALL}")
                self.print_order_details(order)
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Order cancelled{Style.RESET_ALL}")
    
    def place_limit_order_menu(self):
        """Handle limit order placement."""
        print(f"\n{Fore.YELLOW}── Place Limit Order ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        side = self.get_side()
        quantity = self.get_input("Enter quantity: ", float)
        price = self.get_input("Enter limit price: ", float)
        
        print(f"\n{Fore.CYAN}Time in Force options: GTC (Good Till Cancel), IOC (Immediate or Cancel), FOK (Fill or Kill){Style.RESET_ALL}")
        time_in_force = self.get_input("Enter time in force (default: GTC): ").upper() or "GTC"
        
        confirm = self.get_input(f"\nConfirm LIMIT {side} order: {quantity} {symbol} @ ${price}? (yes/no): ").lower()
        
        if confirm == 'yes':
            try:
                order = self.bot.place_limit_order(symbol, side, quantity, price, time_in_force)
                print(f"\n{Fore.GREEN}[OK] Order placed successfully!{Style.RESET_ALL}")
                self.print_order_details(order)
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Order cancelled{Style.RESET_ALL}")
    
    def place_stop_limit_order_menu(self):
        """Handle stop-limit order placement."""
        print(f"\n{Fore.YELLOW}── Place Stop-Limit Order (Bonus Feature) ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        side = self.get_side()
        quantity = self.get_input("Enter quantity: ", float)
        stop_price = self.get_input("Enter stop price (trigger): ", float)
        price = self.get_input("Enter limit price: ", float)
        
        print(f"\n{Fore.CYAN}Time in Force options: GTC (Good Till Cancel), IOC (Immediate or Cancel), FOK (Fill or Kill){Style.RESET_ALL}")
        time_in_force = self.get_input("Enter time in force (default: GTC): ").upper() or "GTC"
        
        confirm = self.get_input(f"\nConfirm STOP-LIMIT {side} order: {quantity} {symbol}? (yes/no): ").lower()
        
        if confirm == 'yes':
            try:
                order = self.bot.place_stop_limit_order(symbol, side, quantity, price, stop_price, time_in_force)
                print(f"\n{Fore.GREEN}[OK] Order placed successfully!{Style.RESET_ALL}")
                self.print_order_details(order)
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Order cancelled{Style.RESET_ALL}")
    
    def check_order_status_menu(self):
        """Handle order status check."""
        print(f"\n{Fore.YELLOW}── Check Order Status ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        order_id = self.get_input("Enter order ID: ", int)
        
        try:
            order = self.bot.get_order_status(symbol, order_id)
            print(f"\n{Fore.GREEN}[OK] Order details retrieved{Style.RESET_ALL}")
            self.print_order_details(order)
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
    
    def view_open_orders_menu(self):
        """Handle viewing open orders."""
        print(f"\n{Fore.YELLOW}── View Open Orders ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (leave empty for all): ", allow_empty=True)
        
        try:
            orders = self.bot.get_open_orders(symbol)
            
            if not orders:
                print(f"\n{Fore.YELLOW}No open orders found{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}[OK] Found {len(orders)} open order(s):{Style.RESET_ALL}\n")
                for i, order in enumerate(orders, 1):
                    print(f"{Fore.CYAN}Order #{i}:{Style.RESET_ALL}")
                    self.print_order_details(order)
                    print()
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
    
    def cancel_order_menu(self):
        """Handle order cancellation."""
        print(f"\n{Fore.YELLOW}── Cancel Order ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        order_id = self.get_input("Enter order ID to cancel: ", int)
        
        confirm = self.get_input(f"\nConfirm cancellation of order {order_id} for {symbol}? (yes/no): ").lower()
        
        if confirm == 'yes':
            try:
                result = self.bot.cancel_order(symbol, order_id)
                print(f"\n{Fore.GREEN}[OK] Order cancelled successfully!{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Cancellation aborted{Style.RESET_ALL}")
    
    def check_balance_menu(self):
        """Handle balance check."""
        print(f"\n{Fore.YELLOW}── Account Balance ──{Style.RESET_ALL}")
        
        try:
            balance = self.bot.get_account_balance()
            print(f"\n{Fore.GREEN}[OK] Balance retrieved:{Style.RESET_ALL}\n")
            
            for asset in balance:
                if float(asset['balance']) > 0:
                    print(f"{Fore.WHITE}  {asset['asset']}: {float(asset['balance']):.8f}")
                    print(f"    Available: {float(asset['availableBalance']):.8f}")
                    print(f"    Unrealized PnL: {float(asset['crossUnPnl']):.8f}")
                    print()
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
    
    def get_current_price_menu(self):
        """Handle current price check."""
        print(f"\n{Fore.YELLOW}── Get Current Price ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        
        try:
            price = self.bot.get_current_price(symbol)
            print(f"\n{Fore.GREEN}[OK] Current price for {symbol}: ${price:,.2f}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
    
    def view_symbol_info_menu(self):
        """Handle symbol info display."""
        print(f"\n{Fore.YELLOW}── Symbol Information ──{Style.RESET_ALL}")
        
        symbol = self.get_input("Enter symbol (e.g., BTCUSDT): ").upper()
        
        try:
            info = self.bot.get_symbol_info(symbol)
            print(f"\n{Fore.GREEN}[OK] Symbol information for {symbol}:{Style.RESET_ALL}\n")
            print(f"{Fore.WHITE}  Status: {info['status']}")
            print(f"  Base Asset: {info['baseAsset']}")
            print(f"  Quote Asset: {info['quoteAsset']}")
            print(f"  Price Precision: {info['pricePrecision']}")
            print(f"  Quantity Precision: {info['quantityPrecision']}")
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR] Error: {str(e)}{Style.RESET_ALL}")
    
    def print_order_details(self, order: dict):
        """Print formatted order details."""
        print(f"{Fore.WHITE}  Symbol: {order.get('symbol')}")
        print(f"  Order ID: {order.get('orderId')}")
        print(f"  Side: {order.get('side')}")
        print(f"  Type: {order.get('type')}")
        print(f"  Status: {order.get('status')}")
        print(f"  Quantity: {order.get('origQty')}")
        print(f"  Executed Qty: {order.get('executedQty')}")
        
        if order.get('price'):
            print(f"  Price: ${order.get('price')}")
        
        if order.get('stopPrice'):
            print(f"  Stop Price: ${order.get('stopPrice')}")
        
        if order.get('avgPrice'):
            print(f"  Avg Price: ${order.get('avgPrice')}")
        
        print(f"  Time in Force: {order.get('timeInForce', 'N/A')}{Style.RESET_ALL}")
    
    def run(self):
        """Run the CLI application."""
        self.print_header()
        
        # Initialize bot
        if not self.initialize_bot():
            print(f"\n{Fore.RED}Failed to initialize bot. Exiting...{Style.RESET_ALL}")
            return
        
        # Main loop
        while self.running:
            try:
                self.print_menu()
                choice = self.get_input("\nEnter your choice (0-9): ")
                
                if choice == '1':
                    self.place_market_order_menu()
                elif choice == '2':
                    self.place_limit_order_menu()
                elif choice == '3':
                    self.place_stop_limit_order_menu()
                elif choice == '4':
                    self.check_order_status_menu()
                elif choice == '5':
                    self.view_open_orders_menu()
                elif choice == '6':
                    self.cancel_order_menu()
                elif choice == '7':
                    self.check_balance_menu()
                elif choice == '8':
                    self.get_current_price_menu()
                elif choice == '9':
                    self.view_symbol_info_menu()
                elif choice == '0':
                    print(f"\n{Fore.GREEN}Thank you for using the Trading Bot. Goodbye!{Style.RESET_ALL}\n")
                    self.running = False
                else:
                    print(f"\n{Fore.RED}[ERROR] Invalid choice. Please enter a number from 0 to 9{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
                confirm = self.get_input("\nDo you want to exit? (yes/no): ").lower()
                if confirm == 'yes':
                    print(f"\n{Fore.GREEN}Goodbye!{Style.RESET_ALL}\n")
                    self.running = False
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Unexpected error: {str(e)}{Style.RESET_ALL}")


def main():
    """Main entry point."""
    cli = TradingBotCLI()
    cli.run()


if __name__ == "__main__":
    main()
