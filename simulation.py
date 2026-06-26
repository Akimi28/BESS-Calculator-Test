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
# DAILY PEAK LOAD
# =====================================================
def get_daily_peak_load_by_daytype(df, day_type, start_hour=14, end_hour=22):
    if day_type not in {"weekday", "weekend"}:
        raise ValueError("day_type must be 'weekday' or 'weekend'")

    df = df.copy()

    if day_type == "weekday":
        df = df[~df["timestamp"].apply(is_weekend)]
    else:
        df = df[df["timestamp"].apply(is_weekend)]

    ts = df["timestamp"]
    df = df[(ts.dt.hour >= start_hour) & (ts.dt.hour < end_hour)]

    df["date"] = df["timestamp"].dt.date

    daily_peak = df.groupby("date", as_index=False)["load_kw"].max()
    daily_peak = daily_peak.rename(columns={"load_kw": "daily_peak_kw"})

    return daily_peak


def get_monthly_highest_daily_peak(df, day_type, start_hour=14, end_hour=22):
    daily_peak = get_daily_peak_load_by_daytype(
        df=df,
        day_type=day_type,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    if daily_peak.empty:
        return pd.DataFrame(
            columns=["month", "highest_daily_peak_kw", "date_of_peak"]
        )

    daily_peak["month"] = (
        pd.to_datetime(daily_peak["date"])
        .dt.to_period("M")
        .astype(str)
    )

    idx = daily_peak.groupby("month")["daily_peak_kw"].idxmax()

    monthly = daily_peak.loc[idx, ["month", "date", "daily_peak_kw"]].copy()
    monthly = monthly.rename(
        columns={
            "date": "date_of_peak",
            "daily_peak_kw": "highest_daily_peak_kw",
        }
    )
    monthly = monthly.sort_values("month")

    return monthly


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

    if time_col is None:
        raise ValueError("Cannot detect timestamp column")

    if load_col is None:
        raise ValueError("Cannot detect load column")

    df = df[[time_col, load_col]]
    df.columns = ["timestamp", "load_kw"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load_kw"] = pd.to_numeric(df["load_kw"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

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
    enable_discharge=True,
):
    max_discharge_kw = qty * unit_kw
    usable_kwh = qty * unit_kwh * dod
    buffer_limit = usable_kwh * buffer_pct
    soc = usable_kwh
    current_day = None

    for _, row in df.iterrows():
        ts = row["timestamp"]
        load = row["load_kw"]

        if is_weekend(ts):
            continue

        if current_day != ts.date():
            current_day = ts.date()
            soc = usable_kwh

        if not (14 <= ts.hour < 22):
            continue

        if enable_discharge and load > target_kw:
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
    enable_discharge=True,
):
    weekday_df = df[~df["timestamp"].apply(is_weekend)]

    if weekday_df.empty:
        raise ValueError("No weekday data found")

    max_load = weekday_df["load_kw"].max()

    low = 0
    high = max_load
    best = max_load

    while high - low > 0.1:
        mid = (low + high) / 2

        safe = test_target(
            df,
            mid,
            qty,
            unit_kw,
            unit_kwh,
            dod,
            buffer_pct,
            enable_discharge,
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
    enable_discharge=True,
):
    weekday_df = df[~df["timestamp"].apply(is_weekend)]
    max_load = weekday_df["load_kw"].max()

    results = []

    for qty in range(1, max_qty + 1):
        if not enable_discharge:
            target = max_load
            shaved_kw = 0
        else:
            target = optimize_target(
                df,
                qty,
                unit_kw,
                unit_kwh,
                dod,
                buffer_pct,
                enable_discharge,
            )

            shaved_kw = max_load - target
            shaved_kw = min(shaved_kw, qty * unit_kw)

        capex = qty * capex_per_unit

        monthly_savings = shaved_kw * savings_per_kw
        annual_savings = monthly_savings * 12

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
