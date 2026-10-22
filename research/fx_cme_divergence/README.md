# FX broker vs CME: two reference rules

Python scripts and CLI only; stock `vectorbt` 0.28.1 accounting; no orders, notebooks or paid data.

* **FROZEN:** take the last CME trade received before the previous session closed. Buy at the broker ask
  if it is sufficiently below the reference; sell at the broker bid if sufficiently above it.
* **INDICATIVE:** compare the broker's executable side with the CME pre-open indication. By default
  (`indicative_crossed_mid: true`) use each futures leg's `(bid + ask) / 2`, then combine the leg mids.
  Pre-open books can cross because matching has not started. Their mid is an indication, **not a CME fill**.

Both references carry the median broker–CME basis measured before close. Sunday carries Friday's basis;
raw and adjusted residuals are retained. CME catching up at reopen is **not** broker convergence.
The EURAUD 18 October check found that increasing reference age alone recovered no INDICATIVE signal;
allowing crossed per-leg mids did. That is a sensitivity, not evidence of robust alpha.

## Rules in force (defaults)

| Rule | Exact test | Default value |
| --- | --- | --- |
| Broker prices | bid and ask present, finite and > 0 | always |
| Broker book | bid < ask | always |
| No look-ahead | quote receive time <= decision time | always |
| Traded-side freshness | decision time - traded-side receive time <= 240 s; ask for a long, bid for a short, and the same side of the USD conversion pair | `traded_side_max_age_s: 240` |
| Other broker side | no freshness requirement | `untraded_side_max_age_s: null` |
| Broker flags / sizes / row age | not tested | `reject_flagged_quotes: false`, `require_quote_size: false`, `broker_row_max_age_s: null` |
| FROZEN reference | last CME trade before the close, at most 300 s old at the close | `frozen_max_age_s: 300` |
| INDICATIVE reference | resting pre-open book received since pre-open and before reopen, at most 3600 s old; per-leg mid, crossed books allowed | `indicative_max_age_s: 3600`, `indicative_crossed_mid: true` |

Entry uses the latest broker quote on the traded side as of signal + 100 ms. Exit uses the latest quote on the
closing side as of reopen + 2 min. Both are accepted while the traded side is within 240 s; there is no wait
for a new update row. A stale quote on the side we do not trade does not block a trade.

## Pairs, units and historical contracts

All **28** conventional pairs among EUR, GBP, AUD, NZD, CAD, CHF, JPY and USD were verified in
`ICMarkets_Risk.symbols` on 2026-10-20. No pair was dropped. The spellings and product terms are in
[`products.py`](products.py); `broker-fetch` rechecks the mapping, adding USD conversion pairs when needed.
`--pairs all` is the default; comma-separated subsets also work. There is **no pair priority**.

All seven futures quote **USD per foreign currency unit**:

| Root | Currency | Units / contract | Outright tick (USD / unit) | Tick USD | Last trade before third Wednesday |
| --- | --- | ---: | ---: | ---: | --- |
| 6E | EUR | 125,000 | .00005 | 6.25 | 2 business days, 09:16 CT |
| 6B | GBP | 62,500 | .0001 | 6.25 | 2 business days, 09:16 CT |
| 6A | AUD | 100,000 | .00005 | 5 | 2 business days, 09:16 CT |
| 6N | NZD | 100,000 | .00005 | 5 | 2 business days, 09:16 CT |
| 6C | CAD | 100,000 | .00005 | 5 | **1** business day, 09:16 CT |
| 6S | CHF | 125,000 | .00005 | 6.25 | 2 business days, 09:16 CT |
| 6J | JPY | 12,500,000 | .0000005 | 6.25 | 2 business days, 09:16 CT |

Official CME `/markets/fx/g10/<product>/specs` URLs are recorded in `products.py`, retrieved 2026-10-20.
USDCAD/USDCHF/USDJPY invert the corresponding future: bid = `1 / ask`, ask = `1 / bid`; available size
also changes base currency. Other crosses use `num.bid / den.ask`, `num.ask / den.bid`. Pip = .01 for
JPY-quoted pairs, .0001 otherwise. INDICATIVE mid mode uses **ratio of leg mids**, not mid of a crossed ratio.

**No current-contract shortcut.** Per root/session, choose the unexpired outright outside the roll buffer,
known by close, with greatest volume in the latest complete prior session available by close. Synthetic
legs must share a tenor. `outputs/audits/contracts.csv` records symbols, IDs, definition knowledge, expiry,
last trade, cutoff, prior volume and roll flags. No back-adjustment. This code-first delivery has **not**
validated the June/September rolls on new real data; inspect that audit before interpreting results.

## Reference settings and sizing

`--config` is a strict YAML mapping; resolved settings are saved with prepared data/results.

| Setting | Default | Meaning |
| --- | --- | --- |
| `traded_side_max_age_s` | 240 | Maximum age of the traded side and the USD conversion side at entry/exit |
| `untraded_side_max_age_s` | null | Optional cap on the other broker side; null means no test |
| `broker_row_max_age_s` | null | Optional whole-row age cap; null means no test |
| `require_quote_size` | false | Optional old size rule for broker and reference rows |
| `reject_flagged_quotes` | false | Optional old indicative/disabled flag rule |
| `frozen_max_age_s` | 300 | Maximum last-trade age **at close**, not its age throughout the closure |
| `indicative_max_age_s` | 3600 | Safety cap on time since a pre-open book update; separate from broker age |
| `indicative_crossed_mid` | true | Combine per-leg mids; crossed pre-open books are allowed |
| `stale_ms` | 2000 | Persistence continuity and preclosure basis CME age only |
| `notional_usd` | 100000 | Fixed USD allocation **per trade** |

Profiles: `configs/pilot.yml` = all pairs, USD 100k, the relaxed defaults above;
`configs/strict_gates.yml` = all pairs, USD 100k, the old behaviour (both broker sides 2 s, flags and size
rules, whole-row 2 s, FROZEN/INDICATIVE 2 s, crossed books rejected) for comparison;
`configs/resting_mid.yml` = all pairs, USD 100k, named reference profile, now equal to the defaults;
`configs/legacy_baseline.yml` = original six pairs, 1,000 base units, old 2 s reference and strict quote gates.
The defaults are a documented rule choice, not a fitted winner.

A resting reference must have been received **and generated** since pre-open began. It is not carried
past reopening. An unexpected close/pause/trading transition during pre-open rejects the session.
Missing/nonpositive prices reject it. CME flags and sizes are not tested by default; locked books are
allowed, and crossed books are allowed in mid mode. Broker books must keep bid < ask; only the traded
side is tested for age, and flags, sizes, the other side and whole-row age are not tested by default.
Rejection labels distinguish age, future receive time, missing timestamps, crossed books and ended pre-open phases.

USD sizing uses the same provider's causal base/USD ask for buys and bid for sells at entry. Missing or
stale conversion on that side makes entry unfillable; broker size is not a default gate. P&L converts quote
currency at exit (bid for gains, ask for liabilities), including reciprocal CAD/CHF/JPY conversions. `quantity` is only a legacy
fixed-base-unit override. All trades use independent, fully funded quote-currency vectorbt accounts:
**no shared capital limit, leverage, netting or preference between pairs**. This is an all-pairs realized
P&L analysis, not a margin/capital-constrained portfolio simulator.

## Install and credentials

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

The pinned private `ICMarkets/TIR_Tools` dependency needs work GitHub auth. Credentials stay outside git
and are read only through explicit flags: `--env-file ~/.config/tir-data/databento.env` and
`--creds-file ~/.config/ch-axi/creds.toml`. Nothing imports a sibling checkout or installs globally.
No-argument invocation identifies the CLI; every command has `--help`; unknown flags fail, no prompts.
Every successful stage ends with `elapsed_s` and `rows` (generated table rows; report counts summary groups).
Estimate/fetch print `rows: null` because they do not decode raw DBN records; request counts are printed instead.
Broadway is the only default provider. ICMLD remains available explicitly with `--providers ICMLD`.

## Reproduce the resolved 180-day run (market-data download is YOUR step)

Resolved on **2026-10-20 UTC**: **2026-04-23 inclusive .. 2026-10-20 exclusive**, ending yesterday.
Definitions/status/trades begin Apr 19 for prior complete-session volume. The code bounds the overall
range to 366 days and splits provider requests at 90 days. MBP-1 uses merged basis-to-reopen session
windows; contract/schema quality is not reduced. ClickHouse reads only Broadway by default. It uses one
`epoch_hour IN (...)` pruning list covering the basis and pre-open-to-boundary windows, plus exact integer-ns
ranges in SQL; closed gaps (including the weekend) are skipped. A count-by-hour query plans one data query
up to **20 million rows and 31 days**; larger requests use at most three row-balanced hour-list slices in parallel,
each with its own connection and Parquet part. All queries are read-only, with two threads and a 30-minute
execution/receive timeout. Arrow batches stream directly to Parquet; there is no full-result DataFrame or CSV.
Each part and the total must match the count, otherwise no valid manifest is published.

```bash
PY=.venv/bin/python
W=/absolute/path/outside/git/fx-180d
C=research/fx_cme_divergence/configs/pilot.yml
COMMON=(--workspace "$W" --start 2026-04-23 --end 2026-10-20 --pairs all)
M=research.fx_cme_divergence
$PY -m $M official-inputs "${COMMON[@]}"
I="$W/data/official"
REQUEST=("${COMMON[@]}" --calendar "$I/calendar.csv" \
  --request-start 2026-04-19T00:00:00Z --request-end 2026-10-20T00:00:00Z \
  --env-file ~/.config/tir-data/databento.env --max-bytes 5000000000)
$PY -m $M estimate "${REQUEST[@]}"     # inspect EVERY estimate in outputs/audits/cost.json
$PY -m $M fetch "${REQUEST[@]}"        # fresh estimates + exact USD 0 gate again
$PY -m $M official-inputs "${COMMON[@]}" --downloads "$W/outputs/audits/downloads.json"
$PY -m $M broker-fetch "${COMMON[@]}" --calendar "$I/calendar.csv" \
  --creds-file ~/.config/ch-axi/creds.toml
# Supply sizes.json yourself: explicit LP size-unit multipliers for every exported pair.
$PY -m $M normalize "${COMMON[@]}" --downloads "$W/outputs/audits/downloads.json" \
  --broker-chunks "$W/data/raw/broker" --calendar "$I/calendar.csv" \
  --contract-terms "$I/terms.csv" --broker-size-units /absolute/path/sizes.json
$PY -m $M audit "${COMMON[@]}" --config "$C"
$PY -m $M prepare "${COMMON[@]}" --config "$C"
$PY -m $M backtest "${COMMON[@]}" --config "$C" --stage baseline
$PY -m $M report "${COMMON[@]}" --stage baseline
```

Broker parts are `data/raw/broker/Broadway_0.parquet` (and `_1`, `_2` only for larger requests).
Normalize writes `data/inputs/bundle/{definitions,volumes,sessions,status,quotes,trades}.parquet`, preserving
integer nanoseconds. Raw Databento `.dbn.zst` files and zero-cost receipts are unchanged. Official calendar,
contract terms, audits and trade/portfolio result tables remain CSV for spreadsheets.
**Old CSV bundles are not readable by this version:** rerun `broker-fetch` for old broker CSV exports, then
rerun `normalize`; there is no CSV fallback. Existing raw Databento caches do not need downloading again.

### Run the full 180 days from the existing sealed bundle (no re-fetch)

The sealed bundle already holds the raw broker quotes (bid/ask receive times, sizes, flags) and CME data.
The broker and reference gates are applied in `prepare`, not in `normalize`, so **no fetch, broker-fetch,
estimate or normalize is needed**. Change only the workspace path; the config and bundle stay as shown.

```bash
PY=.venv/bin/python
M=research.fx_cme_divergence
C=research/fx_cme_divergence/configs/pilot.yml
B=/home/koko/allprojects/firstmate/data/pickle-fx-full-backtest/ws/data/inputs/bundle
W=/absolute/path/outside/git/fx-180d-relaxed   # CHANGE THIS

$PY -m $M prepare --workspace "$W" --inputs "$B" \
  --start 2026-04-23 --end 2026-10-20 --config "$C"
$PY -m $M backtest --workspace "$W" --start 2026-04-23 --end 2026-10-20 \
  --config "$C" --stage baseline
$PY -m $M report --workspace "$W" --stage baseline
```

For the old strict comparison, use `configs/strict_gates.yml` and a **different** workspace:

```bash
OLD=research/fx_cme_divergence/configs/strict_gates.yml
S=/absolute/path/outside/git/fx-180d-strict   # CHANGE THIS
$PY -m $M prepare --workspace "$S" --inputs "$B" \
  --start 2026-04-23 --end 2026-10-20 --config "$OLD"
$PY -m $M backtest --workspace "$S" --start 2026-04-23 --end 2026-10-20 \
  --config "$OLD" --stage baseline
$PY -m $M report --workspace "$S" --stage baseline
```

`official-inputs` uses bundled **public schedule facts**, not downloaded prices. The calendar covers
2026-01-01 through 2026-10-20; all seven CME FX roots have identical schedules. It includes holiday
early closes and records incomplete cycles in `calendar_exclusions.csv`; ambiguous opens are excluded,
not guessed. The official Saturday 17:00 closes are retained for the following Sundays; **status must
confirm them**. Do not enable calendar fallback to force questionable sessions through. Regenerate the
Jan-Apr extension with `python research/fx_cme_divergence/data_access/fetch_cme_calendar.py`; if CME
blocks Python, pass monthly Chrome-fetched JSON files with `--json-dir`.
`terms.csv` is generated offline from your receipted historical definitions, checking each root's CME rule.
Business-day exceptions that disagree are listed in `terms_exclusions.csv`, never silently assigned dates.

`fetch` refuses any positive, NaN, failed or missing estimate, or totals above `--max-bytes`. No credit use
is authorized; a zero estimate projects no charge, not proof of how the account is billed. If the cap binds,
use smaller dated batches/workspaces with the **same** parents/schemas/session windows; do not increase paid
cost authority. Raw cache reuse requires matching request/hash/zero-cost receipts. All real files stay outside git.

## What backtest prints and writes

`ledger.csv` includes every candidate/session outcome; `trades.csv` includes filled entries, including
unpriced open positions. The CLI prints a readable TOON trade table with pair/provider/reference, dates,
signal/entry/exit UTC and Chicago times, side, broker prices, base quantity, USD notional, pips, gross USD,
costs, scenario net USD, cost label, exit reason and hold time. Raw integer ns remain in CSV for audit.
`engine_entry_price`/`engine_exit_price` additionally include any explicitly assumed slippage.

`portfolio.csv` combines **all pairs**, separately for each candidate × provider × Sunday/weekday reference
arm; `by_pair.csv`, `by_week.csv`, `by_month.csv` give the corresponding breakdowns. Providers/reference
rules/parameter alternatives are not pooled into fictitious extra independent trades. Summaries include
signals, unfillable/open/unpriced counts, win rate on priced closes, average pips, average/worst USD trade,
and maximum drawdown of **realized cumulative P&L by exit time**, netting simultaneous exits. They are
not mark-to-market drawdowns. Missing USD valuations suppress totals/curves rather than becoming zeros.
`rejections.csv`, contract/coverage audits and `quote_gates.csv` retain why signals or observations failed.
`report` writes static HTML, the single-pair diagnostic chart and the all-pairs cumulative P&L chart.

Spread is paid by bid/ask fills. Commission/financing default to **unknown**, not asserted zero; results
remain `gross_provisional_unknown_costs`. Only pass measured or clearly labelled assumed costs via
`--commission-bps`, `--slippage-bps`, `--financing-quote`; the ledger records them. No invented cost scenario
is a default. Baseline entries are 10/20/40 pips, exits reopen + 2 min (actual status time/receive availability).
The small separate stages remain `scale` (4), `confirm` (2), `exits` (4); run them only if counts justify it.
Weeks stay whole in chronological 60/20/20 train/validation/test splits. No tuning or robust-alpha claim.

## Limits

* LP size was a constant nominal 100,000 in the old six-pair pilot, not verified firm depth. New pair size
  units need your explicit input. USD 100k can exceed that nominal size, legitimately rejecting entries.
* No new 180-day market data were fetched for this code-first delivery. Estimates are not a performance test;
  normalization/preparation still load Parquet into pandas in memory; peak RAM is not benchmarked.
* Crosses are highly correlated (as are providers if you explicitly select more than Broadway). Clock offsets between LP and CME hosts were
  not measured. Depth is a separate, unimplemented variant; top-of-book only, no paid substitute.
