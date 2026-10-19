# FX broker vs CME futures divergence pilot

Auditable, script-only research (no notebooks, no custom simulator, no orders). Question: when CME FX
futures are closed (Sunday pre-open, weekday maintenance halt), does an LP's FX quote dislocate from the
frozen exchange reference and converge before or after CME reopens?

Pairs: EURUSD AUDUSD NZDUSD EURAUD EURNZD AUDNZD (futures 6E / 6A / 6N). Default window: the last three
*completed* calendar months, resolved and recorded at run time. Accounting is stock `vectorbt` 0.28.1
`Portfolio.from_orders`; `tir_data` (pinned) supplies Databento cost/fetch and read-only ClickHouse.

**Status:** tooling + offline tests + synthetic smoke only. No empirical result is claimed anywhere. Anything
printed from `demo` is labelled SYNTHETIC.

## Install (isolated env; never global)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[test]'   # from the repository root
```

`tir-tools` is a pinned Git dependency (`ICMarkets/TIR_Tools` @ `d77ea84…`, see `pyproject.toml`); the install
needs your normal work GitHub auth. Nothing imports a sibling checkout or edits `sys.path`.

Credentials stay outside the repo and are only read when you pass the file explicitly:

| Need | File (explicit flag) | Content |
| --- | --- | --- |
| Databento | `--env-file ~/.config/tir-data/databento.env` | `DATABENTO_API_KEY=…` |
| ClickHouse | `--creds-file ~/.config/ch-axi/creds.toml` | `[prod]` url/username/password (reused, never copied) |

Invoke from any directory: `python -m research.fx_cme_divergence <command>` (`--help` per command; output is
flat `key: value`, errors are `error:` + exit 1/2, unknown flags exit 2, no prompts).
`--workspace` relocates `data/` and `outputs/` (default: this directory, git-ignored).

## Workflow

| Step | Command | Network |
| --- | --- | --- |
| 0 | `demo` – deterministic synthetic bundle | none |
| 1 | `estimate` – free metadata cost of all four schemas | metadata only |
| 2 | `fetch` – download iff **every** estimate is exactly USD 0 | Databento |
| 3 | `broker-fetch` – bounded hourly read-only LP top-of-book export | ClickHouse |
| 4 | `normalize` – DBN + LP → sealed bundle (needs official calendar + contract terms) | none |
| 5 | `audit` – per-session contract/session audit **first** (`outputs/audits/contracts.csv`) | none |
| 6 | `prepare` – causal basis, references, quote validity, coverage | none |
| 7 | `backtest --stage baseline` – prints the candidate count, then vectorbt | none |
| 8 | `report` – summary CSV, chart, static HTML | none |

Tunables without a flag (latency, staleness, quantity, cash, persistence, hold, exit wait, roll buffer, minimum
basis samples) live in `--config configs/pilot.yml`; strict keys, no credentials, CLI flags override, the resolved
config is written next to results.

Smoke (offline): `demo → audit → prepare → backtest → report` with `--start 2026-07-01 --end 2026-08-01`.

Real run sketch (all paths are yours; start with `estimate`):

```bash
M="python -m research.fx_cme_divergence"
$M estimate --env-file ~/.config/tir-data/databento.env --calendar cal.csv \
    --request-start 2026-06-29T00:00:00Z --request-end 2026-10-01T00:00:00Z
$M fetch    --env-file ~/.config/tir-data/databento.env --calendar cal.csv \
    --request-start 2026-06-29T00:00:00Z --request-end 2026-10-01T00:00:00Z
$M broker-fetch --creds-file ~/.config/ch-axi/creds.toml --providers Broadway
$M normalize --downloads outputs/audits/downloads.json --broker-chunks data/raw/broker \
    --calendar cal.csv --contract-terms terms.csv --broker-size-units sizes.json
$M audit && $M prepare && $M backtest --stage baseline && $M report --stage baseline
```

`--request-start` should precede `--start` by ≥2 days so prior complete-session volume exists for the first
selection; sessions without it are rejected in the audit, not guessed.

## Financial safety

* Authority is **USD 0**. `fetch` first obtains fresh metadata cost for *every* request (definition, status,
  mbp-1, trades, every window); any positive, NaN, missing or failed estimate stops before the downloader. A positive
  `--max-cost` is a usage error. `tir_data`'s own `max_cost=0.0` gate still runs. `--max-bytes` (default 5 GB)
  additionally refuses very large zero-cost pulls.
* Cache reuse needs a matching `*.research.json` receipt (request, sha256, cost 0, pinned commit); foreign or
  edited cache files are refused. Manifests are secret-free.
* Observed 2026-10-20 (metadata only, nothing downloaded): the account's estimate for 6E/6A/6N
  `definition`/`status`/`mbp-1`/`trades` over 2026-07-01→10-01 was USD 0.00 for every schema, with 18.2 GB
  billable `mbp-1`. A zero estimate means no charge is projected; confirm on the Databento billing page that this
  reflects an included plan, not credits. Use `--calendar` so `mbp-1` covers only session windows.

## Protocol (what the code enforces)

* **Contracts first.** Per session and root: definitions known by the close (`known_ns ≤ cutoff`), unexpired and
  outside `roll_buffer_days`, ranked by the volume of the **latest complete prior session published before the
  cutoff**. Expiry is never "front month". Raw symbol, instrument id, expiry, last-trade, cutoff, volume
  evidence and roll flag are in the audit. Synthetic legs must share a tenor. Back-adjusted prices are never used.
* **Sessions.** Official CME schedule rows (source URL + version required) in America/Chicago converted to UTC;
  DST tests included. Status events (close/pre-open/reopen) override calendar times; without status the
  session is rejected unless `--allow-calendar-fallback` (marked `status_verified=False`).
* **Four arms, never pooled:** Sunday/weekday × frozen last-tradable CME trade / indicative pre-open book.
  Frozen CME is a stale reference, not truth; CME catching up at reopen is not broker convergence. Locked or
  crossed books are shown as indicative only and the broker is the only traded side.
* **Basis.** Median broker–futures mid difference over a pre-closure window, frozen through the closure. Raw and
  adjusted residuals are both kept; Sunday sessions carry stale Friday basis (flagged).
* **Crosses.** bid = num.bid / den.ask, ask = num.ask / den.bid; size limited by both legs; futures size is in
  currency units (125 000 / 100 000 per contract). Pip = 0.0001 for all six pairs.
* **Causality.** UTC ns throughout; receive vs event time kept; backward-only as-of; per-side broker quote age;
  rejects flags, crossed/locked, missing, stale, thin quotes (reasons in coverage).
* **Fills.** Long enters ask / exits bid; short enters bid / exits ask; entry ≥ latency after signal; baseline exit
  first valid quote ≥ reopen + 2 min (and not before the reopen message was received) within `exit_wait_s`.
  Missing exits are `unpriced_open`, never zero P&L.
* **Candidates (small, staged, printed before running):** baseline 3 (10/20/40 pips), scale 4 (1/2 bp, 2/3× pre-window
  vol, spread-relative eligibility), confirm 2 (immediate vs persistence), exits 4 (60 s / 300 s fixed, half / full
  convergence with adverse stop). Weeks are assigned 60/20/20 train/validation/test chronologically; nothing is
  tuned on validation/test and no ensemble is fitted.

## Engine assumptions (vectorbt)

Each roundtrip is its own `Portfolio.from_orders` (`cash_sharing=False`, `init_cash` in quote currency,
`direction="both"`, size in base units, `allow_partial=False`), so configurations never share capital or net
positions. Commission → vectorbt `fees`, slippage → `slippage`, financing → exit `fixed_fees`. No margin or leverage
model; P&L is in quote currency and converted to USD with the same LP's contemporaneous USD pair (bid for gains,
ask for losses); an unpriced conversion marks the row. Unknown commission/financing ⇒ `gross_provisional_unknown_costs`.

## Limitations / not yet done

* Real run needs two small hand-authored official inputs: the session calendar CSV (columns of `sessions.csv`,
  from <https://www.cmegroup.com/trading-hours.html> and its holiday schedule) and contract terms
  (`root,raw_symbol,last_trade_ns,tenor,calendar_source`). The tool refuses unsourced ones.
* Status action codes used for close (`CLOSE`/`PAUSE`), pre-open (`PRE_OPEN`) and reopen (`TRADING`) are unverified
  on real data; ambiguous transitions leave a session unverified. Inspect the audit before trusting a session.
* Broker `size` units are unverified; `--broker-size-units` (JSON `{"EURUSD": 1, …}`) is mandatory, not guessed.
* LP data are not Capital.com/Exness customer quotes; Broadway is the intended provider, ICMLD provisional.
* ~13 Sundays per quarter: pilot only, correlated crosses/providers, no robust-alpha claim.
* Depth (MBP-10) is not implemented; top-of-book only.
