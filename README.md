# Binance Futures Testnet Trading Bot

A comprehensive Python trading bot for Binance Futures Testnet with support for market, limit, and stop-limit orders.

## Features

✅ **Core Features:**
- Market orders (Buy/Sell)
- Limit orders (Buy/Sell)
- Stop-Limit orders (Bonus feature)
- Real-time order status tracking
- Account balance monitoring
- Open orders management
- Order cancellation
- Symbol information retrieval
- Current price checking

✅ **Technical Features:**
- Comprehensive logging (console + file)
- Error handling for API exceptions
- Input validation
- Interactive CLI with colored output
- Testnet environment support
- Secure credential management

## Requirements

- Python 3.7+
- Binance Futures Testnet account
- API credentials from Binance Testnet

## Installation

### 1. Clone or Download the Project

```bash
cd assignmet-internshala
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Credentials

#### Get Your API Credentials:
1. Visit [Binance Futures Testnet](https://testnet.binancefuture.com)
2. Register and login
3. Generate API Key and Secret from your account settings

#### Configure Credentials:
Create a `.env` file in the project root:

```bash
copy .env.example .env
```

Edit `.env` and add your credentials:
```
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_api_secret_here
```

**Alternative:** The CLI will prompt you for credentials if not found in `.env`

## Usage

### Running the Trading Bot

```bash
python cli.py
```

### Main Menu Options

```
1. Place Market Order       - Execute immediate buy/sell at market price
2. Place Limit Order        - Place order at specific price
3. Place Stop-Limit Order   - Advanced order with stop trigger (Bonus)
4. Check Order Status       - View details of a specific order
5. View Open Orders         - List all pending orders
6. Cancel Order             - Cancel a pending order
7. Check Account Balance    - View account balances and PnL
8. Get Current Price        - Check current market price
9. View Symbol Info         - Get trading pair information
0. Exit                     - Close the application
```

### Example Workflows

#### Placing a Market Order:
1. Select option `1`
2. Enter symbol: `BTCUSDT`
3. Choose side: `BUY` or `SELL`
4. Enter quantity: `0.001`
5. Confirm the order

#### Placing a Limit Order:
1. Select option `2`
2. Enter symbol: `ETHUSDT`
3. Choose side: `BUY`
4. Enter quantity: `0.01`
5. Enter limit price: `2000`
6. Choose time in force: `GTC` (default)
7. Confirm the order

#### Placing a Stop-Limit Order (Bonus):
1. Select option `3`
2. Enter symbol: `BTCUSDT`
3. Choose side: `SELL`
4. Enter quantity: `0.001`
5. Enter stop price: `45000` (trigger price)
6. Enter limit price: `44900` (execution price)
7. Choose time in force: `GTC`
8. Confirm the order

## Project Structure

```
assignmet-internshala/
├── trading_bot.py          # Core bot implementation
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment file
├── .env                   # Your credentials (create this)
├── logs/                  # Auto-generated log files
│   └── trading_bot_*.log
└── README.md              # This file
```

## Code Structure

### trading_bot.py - BasicBot Class

```python
class BasicBot:
    def __init__(self, api_key, api_secret, testnet=True)
    def place_market_order(symbol, side, quantity)
    def place_limit_order(symbol, side, quantity, price, time_in_force='GTC')
    def place_stop_limit_order(symbol, side, quantity, price, stop_price, time_in_force='GTC')
    def get_order_status(symbol, order_id)
    def cancel_order(symbol, order_id)
    def get_open_orders(symbol=None)
    def get_account_balance()
    def get_current_price(symbol)
    def get_symbol_info(symbol)
```

### Logging

All operations are logged to:
- **Console**: INFO level and above
- **File**: DEBUG level (detailed logs in `logs/` directory)

Log files include:
- Timestamps
- API requests and responses
- Order details
- Error messages

## API Reference

### Binance Futures Testnet
- **Base URL**: `https://testnet.binancefuture.com`
- **Documentation**: [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)

### Order Types Supported

1. **MARKET**: Immediate execution at current market price
2. **LIMIT**: Execute at specified price or better
3. **STOP**: Trigger stop-limit order when stop price is reached (Bonus)

### Time in Force Options

- **GTC** (Good Till Cancel): Order stays active until filled or cancelled
- **IOC** (Immediate or Cancel): Execute immediately, cancel unfilled portion
- **FOK** (Fill or Kill): Execute entire order immediately or cancel

## Error Handling

The bot includes comprehensive error handling for:
- API connection errors
- Invalid credentials
- Insufficient balance
- Invalid symbols or parameters
- Network timeouts
- Order execution failures

All errors are logged with detailed information for debugging.

## Security Notes

⚠️ **Important Security Practices:**
- Never commit `.env` file with real credentials
- Keep API keys secure and never share them
- Use testnet for learning and testing
- Enable IP whitelist on Binance for production
- Use API keys with minimal required permissions

## Testing

### Testing on Binance Testnet:

1. Testnet provides free virtual USDT for testing
2. No real money is involved
3. Practice with different order types safely
4. Test error scenarios without risk

### Recommended Test Scenarios:

- Place small market orders (e.g., 0.001 BTC)
- Test limit orders at different prices
- Practice stop-limit orders
- Check order status and cancellation
- Monitor account balance changes

## Troubleshooting

### Common Issues:

**"Invalid API key"**
- Verify credentials in `.env` file
- Ensure using testnet API keys
- Check for extra spaces in credentials

**"Insufficient balance"**
- Visit testnet website to get free test USDT
- Check account balance using option 7

**"Symbol not found"**
- Ensure symbol format is correct (e.g., `BTCUSDT`)
- Use uppercase for symbol names
- Verify symbol exists on Binance Futures

**"Timestamp error"**
- Check system time synchronization
- Ensure stable internet connection

## Development Notes

### Extending the Bot:

The code is structured for easy extension:
- Add new order types in `trading_bot.py`
- Extend CLI menu in `cli.py`
- Customize logging in `_setup_logger()`
- Add validation functions as needed

### Python-Binance Library:

This project uses the official `python-binance` library:
- Handles authentication
- Manages API requests
- Provides type-safe methods
- Includes error handling

## Requirements Met

✅ **Core Requirements:**
- [x] Python language
- [x] Binance Futures Testnet integration
- [x] Market orders (buy/sell)
- [x] Limit orders (buy/sell)
- [x] Official Binance API (python-binance)
- [x] Command-line interface
- [x] Input validation
- [x] Order details and execution status output
- [x] Comprehensive logging
- [x] Error handling

✅ **Bonus Features:**
- [x] Stop-Limit order type
- [x] Enhanced CLI with colors and formatting
- [x] Detailed logging to files
- [x] Account balance monitoring
- [x] Order management (status, cancel, view)
- [x] Current price checking
- [x] Symbol information retrieval

## License

This project is created for the Internshala hiring process.

## Contact

For questions or issues, please contact the developer.

---

**Happy Trading! (On Testnet! 🚀)**
