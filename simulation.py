import pandas as pd

CHARGE_EFFICIENCY = 0.90
DEFAULT_INTERVAL_HRS = 0.5


def is_weekend(timestamp):
    return timestamp.weekday() >= 5


def detect_interval_hours(df):
    if len(df) < 2:
        return DEFAULT_INTERVAL_HRS

    diff = df["timestamp"].sort_values().diff().dropna()
    if diff.empty:
        return DEFAULT_INTERVAL_HRS

    hours = diff.median().total_seconds() / 3600
    return hours if hours > 0 else DEFAULT_INTERVAL_HRS


def load_and_clean_data(file):
    raw = pd.read_excel(file, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw_text = raw.astype(str)

    header_row = None

    for i in range(len(raw_text)):
        row_text = " ".join(raw_text.iloc[i].values).lower()
        has_time = "date" in row_text or "time" in row_text or "timestamp" in row_text
        has_load = "kw" in row_text or "load" in row_text or "demand" in row_text or "power" in row_text

        if has_time and has_load:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cannot detect header row. Please include Date/Time and Load kW columns.")

    df = raw.copy()
    df.columns = df.iloc[header_row].astype(str)
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    time_candidates = []
    load_candidates = []

    for col in df.columns:
        name = str(col).strip().lower()

        if "date" in name or "time" in name or "timestamp" in name:
            parsed = pd.to_datetime(df[col], errors="coerce")
            time_candidates.append((parsed.notna().sum(), col))

        if "kw" in name or "load" in name or "demand" in name or "power" in name:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid_count = numeric.notna().sum()
            non_zero_count = numeric[numeric > 0].count()
            max_value = numeric.max(skipna=True)

            if pd.isna(max_value):
                max_value = 0

            score = non_zero_count * 1000 + valid_count + max_value
            load_candidates.append((score, col))

    if not time_candidates:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                time_candidates.append((parsed.notna().sum(), col))

    if not load_candidates:
        for col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid_count = numeric.notna().sum()
            non_zero_count = numeric[numeric > 0].count()
            max_value = numeric.max(skipna=True)

            if pd.isna(max_value):
                max_value = 0

            score = non_zero_count * 1000 + valid_count + max_value

            if score > 0:
                load_candidates.append((score, col))

    if not time_candidates:
        raise ValueError("Cannot detect timestamp column.")

    if not load_candidates:
        raise ValueError("Cannot detect load kW column.")

    time_col = sorted(time_candidates, reverse=True)[0][1]
    load_col = sorted(load_candidates, reverse=True)[0][1]

    cleaned = df[[time_col, load_col]].copy()
    cleaned.columns = ["timestamp", "load_kw"]

    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned["load_kw"] = pd.to_numeric(cleaned["load_kw"], errors="coerce")

    cleaned = cleaned.dropna()
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No usable timestamp/load data found after cleaning.")

    if cleaned["load_kw"].max() <= 0:
        raise ValueError("Detected load column contains only zero values.")

    cleaned.attrs["detected_time_column"] = str(time_col)
    cleaned.attrs["detected_load_column"] = str(load_col)

    return cleaned


def filter_peak_window(df, start_hour=14, end_hour=22):
    ts = df["timestamp"]

    return df[
        (ts.dt.hour >= start_hour)
        & (ts.dt.hour < end_hour)
    ].copy()


def get_peak_window_weekdays(df, start_hour=14, end_hour=22):
    weekday_df = df[~df["timestamp"].apply(is_weekend)].copy()
    return filter_peak_window(weekday_df, start_hour, end_hour)


def get_weekday_onpeak_max_load(df, start_hour=14, end_hour=22):
    peak_df = get_peak_window_weekdays(df, start_hour, end_hour)

    if peak_df.empty:
        return 0

    return float(peak_df["load_kw"].max())


def get_daily_peak_load_by_daytype(df, day_type, start_hour=14, end_hour=22):
    if day_type not in ["weekday", "weekend"]:
        raise ValueError("day_type must be weekday or weekend")

    data = df.copy()

    if day_type == "weekday":
        data = data[~data["timestamp"].apply(is_weekend)]
    else:
        data = data[data["timestamp"].apply(is_weekend)]

    data = filter_peak_window(data, start_hour, end_hour)

    if data.empty:
        return pd.DataFrame(columns=["date", "daily_peak_kw"])

    data["date"] = data["timestamp"].dt.date

    daily = data.groupby("date")["load_kw"].max().reset_index()
    daily.columns = ["date", "daily_peak_kw"]

    return daily


def get_monthly_highest_daily_peak(df, day_type, start_hour=14, end_hour=22):
    daily = get_daily_peak_load_by_daytype(
        df=df,
        day_type=day_type,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    if daily.empty:
        return pd.DataFrame(columns=["month", "date_of_peak", "highest_daily_peak_kw"])

    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)

    idx = daily.groupby("month")["daily_peak_kw"].idxmax()

    monthly = daily.loc[idx, ["month", "date", "daily_peak_kw"]].copy()
    monthly.columns = ["month", "date_of_peak", "highest_daily_peak_kw"]

    return monthly.sort_values("month").reset_index(drop=True)


def simulate_target(
    df,
    target_kw,
    qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    enable_opp_charging=False,
    start_hour=14,
    end_hour=22,
):
    interval_hrs = detect_interval_hours(df)

    max_power_kw = qty * unit_kw
    usable_energy_kwh = qty * unit_kwh * dod
    buffer_limit_kwh = usable_energy_kwh * buffer_pct

    if usable_energy_kwh <= 0:
        raise ValueError("Usable battery energy must be greater than zero.")

    weekday_df = df[~df["timestamp"].apply(is_weekend)].copy()

    if weekday_df.empty:
        raise ValueError("No weekday data found.")

    soc_kwh = usable_energy_kwh
    current_day = None

    min_soc_kwh = usable_energy_kwh
    hardest_day = None

    total_discharge_kwh = 0.0
    total_charge_kwh = 0.0

    trace_rows = []

    for _, row in weekday_df.iterrows():
        ts = row["timestamp"]
        load_kw = float(row["load_kw"])

        if current_day != ts.date():
            current_day = ts.date()
            soc_kwh = usable_energy_kwh

        grid_kw_after_bess = load_kw
        discharge_kw = 0.0
        charge_kw = 0.0
        action = "Idle"

        if start_hour <= ts.hour < end_hour:

            if load_kw > target_kw:
                required_kw = load_kw - target_kw
                discharge_kw = min(required_kw, max_power_kw)
                grid_kw_after_bess = load_kw - discharge_kw

                if grid_kw_after_bess > target_kw + 0.0001:
                    return {
                        "safe": False,
                        "bottleneck_timestamp": ts,
                        "bottleneck_reason": "Power Limit",
                        "hardest_day": ts.date(),
                        "min_soc_kwh": round(soc_kwh, 2),
                        "min_soc_pct": round(soc_kwh / usable_energy_kwh * 100, 1),
                        "total_discharge_kwh": round(total_discharge_kwh, 2),
                        "total_charge_kwh": round(total_charge_kwh, 2),
                        "trace": pd.DataFrame(trace_rows),
                    }

                energy_used = discharge_kw * interval_hrs

                if soc_kwh - energy_used < buffer_limit_kwh:
                    return {
                        "safe": False,
                        "bottleneck_timestamp": ts,
                        "bottleneck_reason": "Energy Buffer Limit",
                        "hardest_day": ts.date(),
                        "min_soc_kwh": round(soc_kwh, 2),
                        "min_soc_pct": round(soc_kwh / usable_energy_kwh * 100, 1),
                        "total_discharge_kwh": round(total_discharge_kwh, 2),
                        "total_charge_kwh": round(total_charge_kwh, 2),
                        "trace": pd.DataFrame(trace_rows),
                    }

                soc_kwh -= energy_used
                total_discharge_kwh += energy_used
                action = "Discharge"

            elif enable_opp_charging and load_kw < target_kw and soc_kwh < usable_energy_kwh:
                available_grid_headroom_kw = target_kw - load_kw
                charge_kw = min(available_grid_headroom_kw, max_power_kw)

                charge_energy = charge_kw * interval_hrs * CHARGE_EFFICIENCY
                available_battery_space = usable_energy_kwh - soc_kwh

                if charge_energy > available_battery_space:
                    charge_energy = available_battery_space
                    charge_kw = charge_energy / interval_hrs / CHARGE_EFFICIENCY

                soc_kwh += charge_energy
                total_charge_kwh += charge_energy
                grid_kw_after_bess = load_kw + charge_kw
                action = "Opportunity Charge"

        if soc_kwh < min_soc_kwh:
            min_soc_kwh = soc_kwh
            hardest_day = ts.date()

        trace_rows.append(
            {
                "timestamp": ts,
                "date": ts.date(),
                "load_kw": load_kw,
                "target_kw": target_kw,
                "grid_kw_after_bess": grid_kw_after_bess,
                "battery_kw": discharge_kw - charge_kw,
                "soc_kwh": soc_kwh,
                "soc_pct": soc_kwh / usable_energy_kwh * 100,
                "action": action,
            }
        )

    return {
        "safe": True,
        "bottleneck_timestamp": None,
        "bottleneck_reason": None,
        "hardest_day": hardest_day,
        "min_soc_kwh": round(min_soc_kwh, 2),
        "min_soc_pct": round(min_soc_kwh / usable_energy_kwh * 100, 1),
        "total_discharge_kwh": round(total_discharge_kwh, 2),
        "total_charge_kwh": round(total_charge_kwh, 2),
        "trace": pd.DataFrame(trace_rows),
    }


def optimize_target(
    df,
    qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    enable_opp_charging=False,
    start_hour=14,
    end_hour=22,
):
    peak_df = get_peak_window_weekdays(df, start_hour, end_hour)

    if peak_df.empty:
        raise ValueError("No weekday on-peak data found in the selected time window.")

    max_load = float(peak_df["load_kw"].max())

    low = 0.0
    high = max_load
    best_target = max_load

    while high - low > 0.1:
        trial_target = (low + high) / 2

        result = simulate_target(
            df=df,
            target_kw=trial_target,
            qty=qty,
            unit_kw=unit_kw,
            unit_kwh=unit_kwh,
            dod=dod,
            buffer_pct=buffer_pct,
            enable_opp_charging=enable_opp_charging,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        if result["safe"]:
            best_target = trial_target
            high = trial_target
        else:
            low = trial_target

    final_result = simulate_target(
        df=df,
        target_kw=best_target,
        qty=qty,
        unit_kw=unit_kw,
        unit_kwh=unit_kwh,
        dod=dod,
        buffer_pct=buffer_pct,
        enable_opp_charging=enable_opp_charging,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    return {
        "target_kw": round(best_target, 1),
        "simulation": final_result,
    }


def get_daily_diagnostics(
    df,
    target_kw,
    qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    enable_opp_charging=False,
    start_hour=14,
    end_hour=22,
):
    interval_hrs = detect_interval_hours(df)

    max_power_kw = qty * unit_kw
    usable_energy_kwh = qty * unit_kwh * dod
    buffer_limit_kwh = usable_energy_kwh * buffer_pct

    records = []

    weekday_df = df[~df["timestamp"].apply(is_weekend)].copy()
    weekday_df["date"] = weekday_df["timestamp"].dt.date

    for day, day_df in weekday_df.groupby("date"):
        soc_kwh = usable_energy_kwh
        lowest_soc_kwh = usable_energy_kwh

        discharge_energy_kwh = 0.0
        charge_energy_kwh = 0.0

        peak_window = filter_peak_window(day_df, start_hour, end_hour)

        if peak_window.empty:
            continue

        peak_load_kw = peak_window["load_kw"].max()

        for _, row in peak_window.iterrows():
            load_kw = float(row["load_kw"])

            if load_kw > target_kw:
                discharge_kw = min(load_kw - target_kw, max_power_kw)
                energy_kwh = discharge_kw * interval_hrs

                soc_kwh -= energy_kwh
                discharge_energy_kwh += energy_kwh

            elif enable_opp_charging and load_kw < target_kw and soc_kwh < usable_energy_kwh:
                charge_kw = min(target_kw - load_kw, max_power_kw)
                energy_kwh = charge_kw * interval_hrs * CHARGE_EFFICIENCY

                available_space = usable_energy_kwh - soc_kwh

                if energy_kwh > available_space:
                    energy_kwh = available_space

                soc_kwh += energy_kwh
                charge_energy_kwh += energy_kwh

            if soc_kwh < lowest_soc_kwh:
                lowest_soc_kwh = soc_kwh

        records.append(
            {
                "Date": day,
                "Daily Peak (kW)": round(peak_load_kw, 1),
                "Energy Discharged (kWh)": round(discharge_energy_kwh, 1),
                "Energy Charged (kWh)": round(charge_energy_kwh, 1),
                "Lowest SOC (kWh)": round(lowest_soc_kwh, 1),
                "Lowest SOC (%)": round(lowest_soc_kwh / usable_energy_kwh * 100, 1),
                "Ending SOC (kWh)": round(soc_kwh, 1),
                "Buffer (%)": round(buffer_pct * 100, 1),
                "Buffer Violated": lowest_soc_kwh < buffer_limit_kwh,
            }
        )

    diagnostics = pd.DataFrame(records)

    if diagnostics.empty:
        return diagnostics

    diagnostics = diagnostics.sort_values("Lowest SOC (%)").reset_index(drop=True)
    diagnostics.insert(0, "Rank", range(1, len(diagnostics) + 1))
    diagnostics["Hardest Day"] = False
    diagnostics.loc[0, "Hardest Day"] = True

    return diagnostics


def build_degradation_schedule(
    first_year_savings,
    capex,
    project_years=10,
    annual_degradation_pct=2.0,
):
    rows = []
    cumulative_savings = 0.0

    for year in range(1, project_years + 1):
        capacity_factor = (1 - annual_degradation_pct / 100) ** (year - 1)
        annual_savings = first_year_savings * capacity_factor
        cumulative_savings += annual_savings

        rows.append(
            {
                "Year": year,
                "Capacity Factor (%)": round(capacity_factor * 100, 1),
                "Annual Savings (RM)": round(annual_savings, 0),
                "Cumulative Savings (RM)": round(cumulative_savings, 0),
                "Net Position (RM)": round(cumulative_savings - capex, 0),
            }
        )

    return pd.DataFrame(rows)


def run_bess_matrix(
    df,
    max_qty,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    capex_per_unit,
    savings_per_kw,
    enable_opp_charging=False,
    degradation_pct=2.0,
    project_years=10,
    start_hour=14,
    end_hour=22,
):
    peak_df = get_peak_window_weekdays(df, start_hour, end_hour)

    if peak_df.empty:
        raise ValueError("No weekday on-peak data found in the selected time window.")

    max_load = float(peak_df["load_kw"].max())

    if max_load <= 0:
        raise ValueError("Maximum weekday on-peak load is 0 kW.")

    results = []
    diagnostics = {}
    traces = {}
    degradation = {}

    for qty in range(1, int(max_qty) + 1):
        optimum = optimize_target(
            df=df,
            qty=qty,
            unit_kw=unit_kw,
            unit_kwh=unit_kwh,
            dod=dod,
            buffer_pct=buffer_pct,
            enable_opp_charging=enable_opp_charging,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        target = optimum["target_kw"]
        sim = optimum["simulation"]

        shaved_kw = max_load - target

        if shaved_kw < 0:
            shaved_kw = 0

        shaved_kw = min(shaved_kw, qty * unit_kw)

        capex = qty * capex_per_unit
        monthly_savings = shaved_kw * savings_per_kw
        annual_savings = monthly_savings * 12
        roi = capex / annual_savings if annual_savings > 0 else 999

        daily = get_daily_diagnostics(
            df=df,
            target_kw=target,
            qty=qty,
            unit_kw=unit_kw,
            unit_kwh=unit_kwh,
            dod=dod,
            buffer_pct=buffer_pct,
            enable_opp_charging=enable_opp_charging,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        diagnostics[qty] = daily
        traces[qty] = sim["trace"]

        degradation[qty] = build_degradation_schedule(
            first_year_savings=annual_savings,
            capex=capex,
            project_years=project_years,
            annual_degradation_pct=degradation_pct,
        )

        if not daily.empty:
            hardest = daily.iloc[0]
            hardest_day = hardest["Date"]
            lowest_soc = hardest["Lowest SOC (%)"]
            energy_used = hardest["Energy Discharged (kWh)"]
            energy_charged = hardest["Energy Charged (kWh)"]
        else:
            hardest_day = None
            lowest_soc = 100
            energy_used = 0
            energy_charged = 0

        status = sim["bottleneck_reason"] or "Safe"

        results.append(
            {
                "BESS Qty": qty,
                "Power Rating (kW)": round(qty * unit_kw, 1),
                "Usable Energy (kWh)": round(qty * unit_kwh * dod, 1),
                "Buffer (%)": round(buffer_pct * 100, 1),
                "Buffer Reserve (kWh)": round(qty * unit_kwh * dod * buffer_pct, 1),
                "Max Load (kW)": round(max_load, 1),
                "Target Peak (kW)": round(target, 1),
                "Peak Shaved (kW)": round(shaved_kw, 1),
                "Capex (RM)": round(capex, 0),
                "Monthly Savings (RM)": round(monthly_savings, 0),
                "Annual Savings (RM)": round(annual_savings, 0),
                "ROI (Years)": round(roi, 2),
                "Hardest Day": hardest_day,
                "Lowest SOC (%)": lowest_soc,
                "Energy Used (kWh)": energy_used,
                "Energy Charged (kWh)": energy_charged,
                "Opportunity Charging": "Enabled" if enable_opp_charging else "Disabled",
                "Simulation Status": status,
            }
        )

    results_df = pd.DataFrame(results)

    best = best_system(results_df)

    results_df["Recommendation"] = ""

    results_df.loc[
        results_df["BESS Qty"] == int(best["BESS Qty"]),
        "Recommendation",
    ] = "Recommended"

    results_df.attrs["daily_diagnostics"] = diagnostics
    results_df.attrs["traces"] = traces
    results_df.attrs["degradation"] = degradation
    results_df.attrs["max_load"] = max_load
    results_df.attrs["start_hour"] = start_hour
    results_df.attrs["end_hour"] = end_hour

    return results_df


def best_system(results_df):
    if results_df.empty:
        raise ValueError("No simulation results found.")

    safe_df = results_df[results_df["Simulation Status"] == "Safe"].copy()

    if safe_df.empty:
        safe_df = results_df.copy()

    ranked = safe_df.sort_values(
        by=["ROI (Years)", "Peak Shaved (kW)", "Lowest SOC (%)"],
        ascending=[True, False, False],
    )

    return ranked.iloc[0]


def get_saved_daily_diagnostics(results_df, qty):
    return results_df.attrs.get("daily_diagnostics", {}).get(int(qty), pd.DataFrame())


def get_saved_trace(results_df, qty):
    return results_df.attrs.get("traces", {}).get(int(qty), pd.DataFrame())


def get_saved_degradation(results_df, qty):
    return results_df.attrs.get("degradation", {}).get(int(qty), pd.DataFrame())


def get_peak_shaving_profile(results_df, qty):
    trace = get_saved_trace(results_df, qty)

    if trace.empty:
        return trace

    selected = results_df[results_df["BESS Qty"] == int(qty)]

    if selected.empty:
        return trace

    hardest_day = selected.iloc[0]["Hardest Day"]

    if hardest_day is None or pd.isna(hardest_day):
        return trace

    profile = trace[trace["date"] == hardest_day].copy()
    profile["time"] = profile["timestamp"].dt.strftime("%I:%M %p")

    return profile.reset_index(drop=True)
