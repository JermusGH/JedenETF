import os
import pandas as pd
import warnings
import re
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def normalize_company_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper()

    name = re.sub(r"[^\w\s]", "", name)

    suffixes = [
        r"\bINC\b",
        r"\bCORP\b",
        r"\bCORPORATION\b",
        r"\bLTD\b",
        r"\bLIMITED\b",
        r"\bPLC\b",
        r"\bCOMPANY\b",
        r"\bAG\b",
        r"\bSA\b",
        r"\bNV\b",
        r"\bGROUP\b",
        r"\bHOLDINGS\b",
        r"\bHOLDING\b",
        r"\bSE\b",
        r"\bCLASS\s+[ABC]\b",
        r"\bCL\s+[ABC]\b",
        r"\bCO\b(?!\w)",
    ]

    for suffix in suffixes:
        name = re.sub(suffix, "", name)

    name = " ".join(name.split())

    words = name.split()
    return " ".join(words[:2])


def parse_weight(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace("%", "").replace(",", ".").strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def safe_read_csv(filepath):
    for enc in ["utf-8", "utf-8-sig", "cp1250", "latin-1"]:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return None


def parse_ishares(filepath):
    def _try_parse(df):
        if df is None or df.empty:
            return None
        cols = {str(c).strip(): c for c in df.columns}
        ticker_col = cols.get("Ticker") or cols.get("ISIN") or None
        name_col = cols.get("Name") or None
        weight_col = cols.get("Weight (%)") or cols.get("Weight") or None
        if not (name_col and weight_col):
            return None
        result = df.copy()
        result["weight_clean"] = result[weight_col].apply(parse_weight)
        result["ticker_clean"] = (
            result[ticker_col].astype(str) if ticker_col else "Brak"
        )
        result["name_clean"] = result[name_col]
        result = result[
            result["name_clean"].notna()
            & (result["name_clean"].astype(str).str.strip() != "")
        ]
        return result[["ticker_clean", "name_clean", "weight_clean"]].copy()

    try:
        if filepath.endswith(".csv"):
            for enc in ["utf-8-sig", "utf-8", "cp1250", "latin-1"]:
                try:
                    df_skip = pd.read_csv(filepath, skiprows=2, encoding=enc)
                    parsed = _try_parse(df_skip)
                    if parsed is not None:
                        return parsed
                except Exception:
                    continue
            df_plain = safe_read_csv(filepath)
            return _try_parse(df_plain)
        else:
            for skip in [2, 0]:
                try:
                    df = pd.read_excel(filepath, skiprows=skip)
                    parsed = _try_parse(df)
                    if parsed is not None:
                        return parsed
                except Exception:
                    continue
        return None
    except Exception:
        return None


def parse_vanguard(filepath):
    try:
        df = (
            safe_read_csv(filepath)
            if filepath.endswith(".csv")
            else pd.read_excel(filepath)
        )
        if df is None:
            return None
        cols = {str(c).strip().lower(): c for c in df.columns}

        ticker_col = cols.get("ticker") or cols.get("isin") or None
        name_col = cols.get("holding name") or cols.get("name") or None
        weight_col = cols.get("% of market value") or cols.get("weight") or None

        if not (name_col and weight_col):
            return None

        df["weight_clean"] = df[weight_col].apply(parse_weight)
        df["ticker_clean"] = df[ticker_col].astype(str) if ticker_col else "Brak"
        df["name_clean"] = df[name_col]
        return df[["ticker_clean", "name_clean", "weight_clean"]].copy()
    except Exception:
        return None


def parse_amundi(filepath):
    try:
        df = (
            safe_read_csv(filepath)
            if filepath.endswith(".csv")
            else pd.read_excel(filepath)
        )
        if df is None:
            return None
        cols = {str(c).strip().lower(): c for c in df.columns}

        ticker_col = cols.get("isin code") or cols.get("isin") or None
        name_col = cols.get("name") or None
        weight_col = cols.get("weight") or None

        if not (name_col and weight_col):
            return None

        df["weight_clean"] = df[weight_col].apply(parse_weight)
        df["ticker_clean"] = df[ticker_col].astype(str) if ticker_col else "Brak"
        df["name_clean"] = df[name_col]
        return df[["ticker_clean", "name_clean", "weight_clean"]].copy()
    except Exception:
        return None


def parse_invesco(filepath):
    try:
        df = (
            safe_read_csv(filepath)
            if filepath.endswith(".csv")
            else pd.read_excel(filepath)
        )
        if df is None or df.empty:
            return None

        cols = {str(c).strip().lower(): c for c in df.columns}
        name_col = cols.get("name")
        isin_col = cols.get("isin")
        weight_col = cols.get("weight") or cols.get("weight (%)")

        if not (name_col and weight_col):
            return None

        df["weight_clean"] = df[weight_col].apply(parse_weight)
        df["ticker_clean"] = df[isin_col].astype(str) if isin_col else "Brak"
        df["name_clean"] = df[name_col]
        df = df[
            df["name_clean"].notna() & (df["name_clean"].astype(str).str.strip() != "")
        ]
        return df[["ticker_clean", "name_clean", "weight_clean"]].copy()
    except Exception:
        return None


def parse_xtrackers(filepath):
    try:
        df = (
            safe_read_csv(filepath)
            if filepath.endswith(".csv")
            else pd.read_excel(filepath)
        )
        if df is None or df.empty:
            return None

        cols = {str(c).strip().lower(): c for c in df.columns}

        name_col = cols.get("name")
        isin_col = cols.get("isin")
        weight_col = cols.get("weighting") or cols.get("weight") or cols.get("weight (%)")

        if not (name_col and weight_col):
            return None

        df["weight_clean"] = df[weight_col].apply(parse_weight)
        df["ticker_clean"] = df[isin_col].astype(str) if isin_col else "Brak"
        df["name_clean"] = df[name_col]
        df = df[
            df["name_clean"].notna() & (df["name_clean"].astype(str).str.strip() != "")
        ]
        return df[["ticker_clean", "name_clean", "weight_clean"]].copy()
    except Exception:
        return None


def identify_and_parse(filepath):
    try:
        head_parts = []
        if filepath.endswith(".csv"):
            for enc in ["utf-8-sig", "utf-8", "cp1250", "latin-1"]:
                try:
                    df_meta = pd.read_csv(filepath, nrows=1, header=None, encoding=enc)
                    head_parts.append(str(df_meta.values.tolist()).lower())
                    df_skip = pd.read_csv(filepath, skiprows=2, nrows=1, encoding=enc)
                    head_parts.append(str(df_skip.columns.tolist()).lower())
                    break
                except Exception:
                    continue
            df_plain = safe_read_csv(filepath)
            if df_plain is not None:
                head_parts.append(str(df_plain.columns.tolist()).lower())
        else:
            for skip in [0, 2]:
                try:
                    df = pd.read_excel(filepath, nrows=3, skiprows=skip)
                    head_parts.append(str(df.columns.tolist()).lower())
                except Exception:
                    continue

        head = " ".join(head_parts)

        if "fund holdings" in head or "ishares" in head or "weight (%)" in head:
            return parse_ishares(filepath)
        elif "weighting" in head and "type of security" in head:
            return parse_xtrackers(filepath)
        elif "amundi" in head or "isin code" in head:
            return parse_amundi(filepath)
        elif (
            "% of market value" in head or "holding name" in head or "vanguard" in head
        ):
            return parse_vanguard(filepath)
        else:
            for parser in [
                parse_ishares,
                parse_vanguard,
                parse_amundi,
                parse_invesco,
                parse_xtrackers,
            ]:
                res = parser(filepath)
                if res is not None and not res.empty:
                    return res
            return None
    except Exception:
        return None


def fetch_prices_pln(tickers_list):
    if not tickers_list:
        return {}
    if yf is None:
        print("\n  [!] Brak biblioteki 'yfinance'. Zainstaluj: pip install yfinance")
        return {}

    print(f"\n  Pobieranie cen z Yahoo Finance dla: {tickers_list} ...")

    try:
        data = yf.download(tickers_list, period="5d", auto_adjust=True, progress=False)[
            "Close"
        ]
    except Exception as e:
        print(f"  [!] Błąd pobierania danych: {e}")
        return {}

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])

    last_prices = data.dropna(how="all").iloc[-1]

    currencies = {}
    fx_rates = {"PLN": 1.0}

    for t in tickers_list:
        try:
            ticker_obj = yf.Ticker(t)
            curr = ticker_obj.info.get("currency", "PLN").upper()
            currencies[t] = curr

            if curr != "PLN" and curr not in fx_rates:
                fx_ticker = f"{curr}PLN=X"
                fx_data = yf.download(
                    fx_ticker, period="5d", auto_adjust=True, progress=False
                )["Close"]
                fx_data = fx_data.dropna()
                if not fx_data.empty:
                    fx_rates[curr] = float(fx_data.squeeze().iloc[-1])
                    print(f"  Kurs {curr}/PLN: {fx_rates[curr]:.4f}")
                else:
                    print(
                        f"  [!] Nie udało się pobrać kursu {curr}/PLN — przyjmuję 1.0"
                    )
                    fx_rates[curr] = 1.0
        except Exception as e:
            print(f"  [!] Błąd dla tickera {t}: {e}")
            currencies[t] = "PLN"

    result = {}
    for t in tickers_list:
        try:
            price = float(last_prices[t])
            curr = currencies.get(t, "PLN")
            fx = fx_rates.get(curr, 1.0)
            price_pln = round(price * fx, 2)
            result[t] = price_pln
            print(f"  {t}: {price:.2f} {curr} × {fx:.4f} = {price_pln:.2f} PLN")
        except Exception as e:
            print(f"  [!] Brak ceny dla {t}: {e}")
            result[t] = None

    return result


def main():
    folder_path = "ETF exports"
    config_path = "config.csv"

    if not os.path.exists(config_path):
        print(
            f"Błąd: Nie znaleziono pliku '{config_path}' w głównym folderze z programem."
        )
        print("Wymagane kolumny: ETF, Ticker, Q")
        print(
            "  ETF    - nazwa ETF (musi być częścią nazwy pliku w folderze ETF exports)"
        )
        print("  Ticker - ticker Yahoo Finance (np. CSPX.L, VWCE.DE, ISAC.L)")
        print("  Q      - liczba jednostek (quantity)")
        return

    if not os.path.exists(folder_path):
        print(
            f"Utworzono folder '{folder_path}'. Wrzuć tam swoje pliki z portfelami i uruchom ponownie."
        )
        os.makedirs(folder_path)
        return

    files = [
        f for f in os.listdir(folder_path) if f.endswith(".csv") or f.endswith(".xlsx")
    ]
    if not files:
        print(f"Folder '{folder_path}' jest pusty.")
        return

    try:
        config_df = pd.read_csv(config_path)
        config_df.columns = [c.strip() for c in config_df.columns]
    except Exception as e:
        print(f"Błąd wczytywania pliku config.csv: {e}")
        return

    required_cols = {"ETF", "Ticker", "Q"}
    missing = required_cols - set(config_df.columns)
    if missing:
        print(f"Błąd: Brakuje kolumn w config.csv: {missing}")
        print(
            "Kolumna 'P' (cena) nie jest już wymagana - ceny pobierane są automatycznie z Yahoo Finance."
        )
        return

    tickers_list = config_df["Ticker"].dropna().astype(str).str.strip().tolist()
    tickers_list = [t for t in tickers_list if t and t != "nan"]

    prices_pln = fetch_prices_pln(tickers_list)

    portfolio = []
    total_portfolio_value = 0

    print("\n--- KREATOR ZUNIFIKOWANEGO ETF (Smart Merge) ---")

    for f in files:
        filepath = os.path.join(folder_path, f)
        filename_no_ext = os.path.splitext(f)[0]

        def etf_matches(etf_name, filename):
            etf_clean = str(etf_name).strip()
            return (
                filename.startswith(etf_clean)
                or filename == etf_clean
                or etf_clean in filename
            )

        match = config_df[
            config_df["ETF"].apply(lambda x: etf_matches(x, filename_no_ext))
        ]
        if match.empty:
            continue

        etf_full_name = str(match["ETF"].iloc[0]).strip()
        ticker_yf = str(match["Ticker"].iloc[0]).strip()
        units = float(match["Q"].iloc[0])

        if units <= 0:
            continue

        price = prices_pln.get(ticker_yf)
        if price is None:
            print(f"\n  [!] Brak ceny dla {ticker_yf} — pomijam {f}")
            continue

        print(
            f"\nPlik: {f} | ETF: {etf_full_name} | Ticker: {ticker_yf} | "
            f"Cena: {price:.2f} PLN | Wartość: {units * price:.2f} PLN"
        )

        etf_total_value = units * price
        df = identify_and_parse(filepath)

        if df is not None and not df.empty:
            if df["weight_clean"].sum() > 10:
                df["value_in_portfolio"] = (df["weight_clean"] / 100) * etf_total_value
            else:
                df["value_in_portfolio"] = df["weight_clean"] * etf_total_value

            df["etf_source"] = ticker_yf
            df["merge_key"] = df["name_clean"].apply(normalize_company_name)

            total_portfolio_value += etf_total_value
            portfolio.append(df)
        else:
            print("  [!] Nie udało się rozpoznać struktury pliku.")

    if not portfolio:
        print("\nBrak poprawnych danych do analizy.")
        return

    combined_df = pd.concat(portfolio, ignore_index=True)

    combined_df["ticker_len"] = combined_df["ticker_clean"].astype(str).apply(len)
    combined_df = combined_df.sort_values("ticker_len")

    grouped = (
        combined_df.groupby("merge_key")
        .agg(
            {
                "ticker_clean": "first",
                "name_clean": "first",
                "value_in_portfolio": "sum",
            }
        )
        .reset_index()
    )

    grouped["unified_weight_%"] = (
        grouped["value_in_portfolio"] / total_portfolio_value
    ) * 100

    combined_df["contrib_weight_%"] = (
        combined_df["value_in_portfolio"] / total_portfolio_value
    ) * 100
    pivot = combined_df.pivot_table(
        index="merge_key",
        columns="etf_source",
        values="contrib_weight_%",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    final_df = pd.merge(grouped, pivot, on="merge_key")
    top20 = final_df.sort_values(by="unified_weight_%", ascending=False).head(20)

    etf_sources = list(pivot.columns)
    etf_sources.remove("merge_key")

    results_csv = "unified_etf_results.csv"
    summary_txt = "unified_etf_summary.txt"
    final_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"\nZapisano pełną tabelę wyników: {os.path.abspath(results_csv)}")

    W_TICKER = max(len("Ticker / ISIN"), 12)
    for _, r in top20.iterrows():
        tl = len(str(r["ticker_clean"]).replace("nan", "Brak").strip())
        W_TICKER = max(W_TICKER, tl)
    W_TICKER = min(W_TICKER, 24)
    hdr_name = "Nazwa (Spółka)"
    W_NAME = len(hdr_name)
    for _, r in top20.iterrows():
        nm = str(r["name_clean"]).strip()
        if nm and nm != "nan":
            W_NAME = max(W_NAME, len(nm))
    W_NAME = min(W_NAME, 40)
    W_WAGA = 9
    W_VALUE = 12
    W_SRC = max(9, max((len(str(s)) for s in etf_sources), default=0))
    W_SRC = min(W_SRC, 18)

    def cell_ticker(raw):
        s = str(raw).replace("nan", "Brak").strip()
        if len(s) > W_TICKER:
            s = s[: W_TICKER - 1] + "…"
        return s.ljust(W_TICKER)

    def cell_name(raw):
        if pd.isna(raw):
            s = ""
        else:
            s = str(raw).strip()
        if s == "nan":
            s = ""
        if len(s) > W_NAME:
            s = s[: W_NAME - 1] + "…"
        return s.ljust(W_NAME)

    def hdr_src(s):
        s = str(s)
        if len(s) > W_SRC:
            s = s[: W_SRC - 1] + "…"
        return s.ljust(W_SRC)

    def cell_waga(x):
        return f"{x:>6.2f}%".rjust(W_WAGA)

    def cell_value(x):
        return f"{x:>{W_VALUE}.2f}"

    def cell_src(x):
        if x > 0.005:
            return f"{x:>6.2f}%".rjust(W_SRC)
        return f"{'-':>{W_SRC}}"

    hdr = (
        f"{'Ticker / ISIN':<{W_TICKER}} | {hdr_name:<{W_NAME}} | {'Waga (%)':>{W_WAGA}} | "
        f"{'Wartość PLN':>{W_VALUE}} | " + " | ".join(hdr_src(s) for s in etf_sources)
    )
    sep = "-" * len(hdr)

    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    emit()
    emit()
    emit("=" * len(hdr))
    emit("TOP 20 POZYCJI W TWOIM ZUNIFIKOWANYM PORTFELU:")
    emit("=" * len(hdr))
    emit(hdr)
    emit(sep)

    for _, row in top20.iterrows():
        t = cell_ticker(row["ticker_clean"])
        n = cell_name(row["name_clean"])
        w = cell_waga(row["unified_weight_%"])
        v = cell_value(row["value_in_portfolio"])
        src_cells = " | ".join(cell_src(row[src]) for src in etf_sources)
        emit(f"{t} | {n} | {w} | {v} | {src_cells}")

    emit(sep)
    emit(f"Całkowita wartość portfela: {total_portfolio_value:.2f} PLN")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Zapisano podsumowanie (TOP 20): {os.path.abspath(summary_txt)}")


if __name__ == "__main__":
    main()
