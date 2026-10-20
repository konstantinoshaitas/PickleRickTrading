"""Verified CME outrights, all quoted in USD per foreign currency unit.

Official contract specs retrieved 2026-10-20. Broker pair spellings were checked
against ICMarkets_Risk.symbols that day; fetch rechecks the mapping on every run.
"""
# currency: (root, foreign units/contract, outright tick in USD, business days before third Wednesday, slug)
PRODUCTS = {
    "EUR": ("6E", 125000, 0.00005, 2, "euro-fx"),
    "GBP": ("6B", 62500, 0.0001, 2, "british-pound"),
    "AUD": ("6A", 100000, 0.00005, 2, "australian-dollar"),
    "NZD": ("6N", 100000, 0.00005, 2, "new-zealand-dollar"),
    "USD": None,
    "CAD": ("6C", 100000, 0.00005, 1, "canadian-dollar"),
    "CHF": ("6S", 125000, 0.00005, 2, "swiss-franc"),
    "JPY": ("6J", 12500000, 0.0000005, 2, "japanese-yen"),
}
ROOTS = {p[0]: c for c, p in PRODUCTS.items() if p}
MULTIPLIER = {p[0]: p[1] for p in PRODUCTS.values() if p}
SOURCES = {p[0]: f"https://www.cmegroup.com/markets/fx/g10/{p[4]}/specs"
           for p in PRODUCTS.values() if p}
SUPPORTED_PAIRS = (
    "EURGBP", "EURAUD", "EURNZD", "EURUSD", "EURCAD", "EURCHF", "EURJPY",
    "GBPAUD", "GBPNZD", "GBPUSD", "GBPCAD", "GBPCHF", "GBPJPY",
    "AUDNZD", "AUDUSD", "AUDCAD", "AUDCHF", "AUDJPY",
    "NZDUSD", "NZDCAD", "NZDCHF", "NZDJPY", "USDCAD", "USDCHF", "USDJPY", "CADCHF", "CADJPY", "CHFJPY",
)
LEGS = {pair: tuple(PRODUCTS[c][0] for c in (pair[:3], pair[3:]) if PRODUCTS[c])
        for pair in SUPPORTED_PAIRS}
PIP = {pair: 0.01 if pair.endswith("JPY") else 0.0001 for pair in SUPPORTED_PAIRS}


def broker_pairs(pairs):
    """Include the causal USD crosses needed to size entries and price exits."""
    currencies = {pair[:3] for pair in pairs} | {pair[3:] for pair in pairs}
    conversions = {quote_in_usd(c)[0] for c in currencies if c != "USD"}
    return tuple(sorted(set(pairs) | conversions))


def quote_in_usd(currency):
    """Broker USD conversion pair, and whether its bid/ask must be reciprocated."""
    direct = currency+"USD"
    return (direct, False) if direct in SUPPORTED_PAIRS else ("USD"+currency, True)
