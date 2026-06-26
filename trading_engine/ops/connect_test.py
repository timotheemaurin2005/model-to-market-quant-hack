"""
connect_test.py — confirm the local MT5 terminal can attach to the
competition server in browsing mode. Places NO orders.

Credentials are never hardcoded: read from MT5_LOGIN/MT5_SERVER/MT5_PASSWORD env vars.

Usage:
    MT5_LOGIN='...' MT5_SERVER='...' MT5_PASSWORD='...' python connect_test.py
"""

import os
import sys

import MetaTrader5 as mt5


def main():
    login = os.environ.get("MT5_LOGIN")
    server = os.environ.get("MT5_SERVER")
    password = os.environ.get("MT5_PASSWORD")
    if not login or not server or not password:
        print("MT5_LOGIN, MT5_SERVER, and MT5_PASSWORD env vars must all be set. Run with:")
        print("  MT5_LOGIN='...' MT5_SERVER='...' MT5_PASSWORD='...' python connect_test.py")
        sys.exit(1)

    ok = mt5.initialize(login=int(login), server=server, password=password)
    if not ok:
        print(f"initialize() failed, last_error = {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    print("Connected.")
    print("terminal_info():", mt5.terminal_info())
    print("account_info():", mt5.account_info())

    mt5.shutdown()


if __name__ == "__main__":
    main()
