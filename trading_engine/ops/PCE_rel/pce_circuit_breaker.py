"""
pce_circuit_breaker.py — The Red Line Guardrail
PURPOSE: If total open P&L hits -$25,000, flatten ALL positions immediately.
         Preserves competition rank by preventing catastrophic drawdown.
USAGE: Run in a separate terminal alongside directional_sleeve.py
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
HARD_STOP_USD = -25_000.0   # Trigger level
CHECK_INTERVAL = 1           # Seconds between checks
MAX_FLATTEN_ATTEMPTS = 3     # Retry attempts to close positions


# ── FLATTEN ALL POSITIONS ─────────────────────────────────────
def flatten_all():
    """
    Close every open position. Retries up to MAX_FLATTEN_ATTEMPTS times.
    """
    print("\n" + "="*50)
    print("🚨 CIRCUIT BREAKER TRIGGERED!")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    print("🚨 FLATTENING ALL POSITIONS...")
    print("="*50)

    for attempt in range(1, MAX_FLATTEN_ATTEMPTS + 1):
        positions = mt5.positions_get()

        if not positions:
            print(f"✅ All positions successfully closed (attempt {attempt})")
            return True

        print(f"   Attempt {attempt}: {len(positions)} positions remaining...")

        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                print(f"   ⚠️ No tick for {pos.symbol} — skipping")
                continue

            # Closing order is opposite to position type
            if pos.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                close_price = tick.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                close_price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 50,  # Wide deviation for emergency close
                "type_filling": mt5.ORDER_FILLING_IOC,
                "comment": "CIRCUIT_BREAKER_KILL",
            }

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"   ✅ Closed {pos.symbol} | Ticket: {pos.ticket} | P&L: ${pos.profit:.2f}")
            else:
                print(f"   ❌ Failed {pos.symbol} | Code: {result.retcode} | {result.comment}")

        time.sleep(1)

    # Final check
    remaining = mt5.positions_get()
    if remaining:
        print(f"⚠️ WARNING: {len(remaining)} positions could not be closed!")
        for pos in remaining:
            print(f"   - {pos.symbol} | {pos.volume} lots | P&L: ${pos.profit:.2f}")
        return False

    return True


# ── MONITORING LOOP ───────────────────────────────────────────
def monitor():
    """
    Continuously monitor total P&L.
    Triggers flatten_all() if loss exceeds HARD_STOP_USD.
    """
    print("🛡️  CIRCUIT BREAKER ARMED")
    print(f"   Hard stop: ${HARD_STOP_USD:,.0f}")
    print(f"   Check interval: {CHECK_INTERVAL}s")
    print(f"   Account: {mt5.account_info().login}")
    print("-" * 50)

    peak_pnl = 0.0

    try:
        while True:
            positions = mt5.positions_get()

            if positions:
                total_pnl = sum(p.profit for p in positions)
                peak_pnl = max(peak_pnl, total_pnl)

                # Status update every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] P&L: ${total_pnl:,.2f} | Peak: ${peak_pnl:,.2f} | Positions: {len(positions)}")

                # Circuit breaker check
                if total_pnl <= HARD_STOP_USD:
                    success = flatten_all()
                    if success:
                        print("✅ Circuit breaker executed successfully. Exiting.")
                    else:
                        print("⚠️ Partial close — manual intervention required!")
                    break

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n⛔ Circuit breaker stopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        exit(1)

    monitor()