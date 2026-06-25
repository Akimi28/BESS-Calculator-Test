import pandas as pd

INTERVAL_HRS = 0.5


# =====================================================
# CALENDAR HELPER
# =====================================================
def is_weekend(timestamp):
    """
    Saturday/Sunday = weekend
    Monday = 0 ... Sunday = 6
    """
    return timestamp.weekday() >= 5


# =====================================================
# LOAD DATA
# =====================================================
def load_and_clean_data(file):

    df = pd.read_excel(file, header=None)
    df = df.dropna(how="all")

    df_str = df.astype(str)

    header_row = None

    for i in range(len(df_str)):
        row = " ".join(df_str.iloc[i].values).lower()

        if ("date" in row or "time" in row) and ("kw" in row or "load" in row):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cannot detect header row")

    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    time_col = None
    load_col = None

    for c in df.columns:
        c_str = str(c).lower()

        if "date" in c_str or "time" in c_str:
            time_col = c

        if "kw" in c_str or "load" in c_str:
            load_col = c

    df = df[[time_col, load_col]]
    df.columns = ["timestamp", "load_kw"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load_kw"] = pd.to_numeric(df["load_kw"], errors="coerce")

    df = df.dropna()

    return df


# =====================================================
# TEST ONE TARGET
# =====================================================
def test_target(
    df,
    target_kw,
    qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
):

    max_discharge_kw = qty * unit_kw
    usable_kwh = qty * unit_kwh * dod
    buffer_limit = usable_kwh * buffer_pct
    soc = usable_kwh

    for _, row in df.iterrows():

        ts = row["timestamp"]
        load = row["load_kw"]

        # skip weekend
        if is_weekend(ts):
            continue

        # reset daily at 14:00
        if ts.hour == 14 and ts.minute == 0:
            soc = usable_kwh

        if not (14 <= ts.hour < 22):
            continue

        if load > target_kw:

            required_kw = load - target_kw

            discharge_kw = min(required_kw, max_discharge_kw)

            actual_grid = load - discharge_kw

            if actual_grid > target_kw:
                return False

            energy_used = discharge_kw * INTERVAL_HRS

            if soc - energy_used <= buffer_limit:
                return False

            soc -= energy_used

    return True


# =====================================================
# FIND LOWEST SAFE TARGET
# =====================================================
def optimize_target(
    df,
    qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
):

    weekday_df = df[~df["timestamp"].apply(is_weekend)]
    max_load = weekday_df["load_kw"].max()

    low = 0
    high = max_load
    best = max_load

    while high - low > 1:

        mid = (low + high) / 2

        safe = test_target(
            df,
            mid,
            qty,
            unit_kw,
            unit_kwh,
            dod,
            buffer_pct
        )

        if safe:
            best = mid
            high = mid
        else:
            low = mid

    return round(best, 1)


# =====================================================
# RUN MATRIX
# =====================================================
def run_bess_matrix(
    df,
    max_qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    capex_per_unit,
    savings_per_kw,
):

    weekday_df = df[~df["timestamp"].apply(is_weekend)]
    max_load = weekday_df["load_kw"].max()

    results = []

    for qty in range(1, max_qty + 1):

        target = optimize_target(
            df,
            qty,
            unit_kw,
            unit_kwh,
            dod,
            buffer_pct
        )

        shaved_kw = max_load - target
        shaved_kw = min(shaved_kw, qty * unit_kw)

        capex = qty * capex_per_unit

        annual_savings = shaved_kw * savings_per_kw
        monthly_savings = annual_savings / 12

        roi = capex / annual_savings if annual_savings > 0 else 999

        results.append({
            "BESS Qty": qty,
            "Power Rating (kW)": qty * unit_kw,
            "Usable Energy (kWh)": round(qty * unit_kwh * dod, 1),
            "Target Peak (kW)": round(target, 1),
            "Peak Shaved (kW)": round(shaved_kw, 1),
            "Capex (RM)": round(capex, 0),
            "Monthly Savings (RM)": round(monthly_savings, 0),
            "ROI (Years)": round(roi, 2),
        })

    return pd.DataFrame(results)


# =====================================================
# BEST SYSTEM
# =====================================================
def best_system(results_df):

    return results_df.sort_values("ROI (Years)").iloc[0]