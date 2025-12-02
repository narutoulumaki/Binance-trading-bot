"""
Example script demonstrating how to use the BasicBot class programmatically.
This shows how to integrate the bot into your own scripts.
"""

import os
from dotenv import load_dotenv
from trading_bot import BasicBot

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    # Initialize the bot
    print("Initializing bot...")
    bot = BasicBot(api_key, api_secret, testnet=True)
    
    # Example 1: Get current price
    print("\n--- Example 1: Get Current Price ---")
    symbol = "BTCUSDT"
    price = bot.get_current_price(symbol)
    print(f"Current {symbol} price: ${price:,.2f}")
    
    # Example 2: Check account balance
    print("\n--- Example 2: Check Account Balance ---")
    balance = bot.get_account_balance()
    for asset in balance:
        if float(asset['balance']) > 0:
            print(f"{asset['asset']}: {float(asset['balance']):.8f}")
    
    # Example 3: Place a limit order (commented out for safety)
    print("\n--- Example 3: Place Limit Order (Demo) ---")
    print("Uncomment the following code to actually place an order:")
    print("""
    # order = bot.place_limit_order(
    #     symbol="BTCUSDT",
    #     side="BUY",
    #     quantity=0.001,
    #     price=30000.00
    # )
    # print(f"Order placed: {order['orderId']}")
    """)
    
    # Example 4: View open orders
    print("\n--- Example 4: View Open Orders ---")
    open_orders = bot.get_open_orders()
    print(f"Found {len(open_orders)} open order(s)")
    for order in open_orders:
        print(f"  Order {order['orderId']}: {order['side']} {order['origQty']} {order['symbol']} @ ${order.get('price', 'MARKET')}")
    
    # Example 5: Get symbol info
    print("\n--- Example 5: Symbol Information ---")
    info = bot.get_symbol_info("BTCUSDT")
    print(f"Symbol: {info['symbol']}")
    print(f"Status: {info['status']}")
    print(f"Price Precision: {info['pricePrecision']}")
    print(f"Quantity Precision: {info['quantityPrecision']}")
    
    print("\n✓ Examples completed!")

if __name__ == "__main__":
    main()
