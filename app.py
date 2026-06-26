import streamlit as st
import pandas as pd
import calendar
import math

from simulation import (
    load_and_clean_data,
    run_bess_matrix,
    best_system,
    get_monthly_highest_daily_peak,
)

st.set_page_config(
    page_title="BESS Proposal Generator",
    layout="wide"
)

st.title("🔋 BESS Peak Shaving Proposal Generator")

uploaded_file = st.file_uploader(
    "Upload Load Profile Excel",
    type=["xlsx"]
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("System Parameters")

unit_kw = st.sidebar.number_input(
    "kW per Container",
    value=125
)

unit_kwh = st.sidebar.number_input(
    "kWh per Container",
    value=261
)

dod = st.sidebar.number_input(
    "DoD",
    min_value=0.0,
    max_value=1.0,
    value=0.95
)

buffer_pct = st.sidebar.number_input(
    "Safety Buffer",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

max_qty = st.sidebar.number_input(
    "Maximum Containers",
    min_value=1,
    value=5
)

enable_discharge = st.sidebar.toggle(
    "Enable Battery Discharge",
    value=True
)

# =====================================================
# FINANCIAL PARAMETERS
# =====================================================

st.sidebar.header("Financial Parameters")

capex_per_unit = st.sidebar.number_input(
    "Capex per Container (RM)",
    value=200000,
    step=1000
)

savings_per_kw = st.sidebar.number_input(
    "Peak Shaving Cost per kW (RM/month)",
    value=97.06
)

if uploaded_file:

    try:
        df = load_and_clean_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    st.success("Load profile imported successfully")

    total_rows = len(df)

    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()

    weekday_df = df[df["timestamp"].dt.weekday < 5]
    weekend_df = df[df["timestamp"].dt.weekday >= 5]

    weekday_days = weekday_df["timestamp"].dt.date.nunique()
    weekend_days = weekend_df["timestamp"].dt.date.nunique()

    st.subheader("📊 Load Profile Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Rows", total_rows)
    c2.metric("Weekdays", weekday_days)
    c3.metric("Weekends", weekend_days)
    c4.metric(
        "Months",
        len(df["timestamp"].dt.to_period("M").unique())
    )

    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {end_date}")

    st.subheader("📄 Full Data Preview (Paginated)")

    preview_df = df.copy()

    preview_df["Day Type"] = preview_df["timestamp"].apply(
        lambda x: "Weekend" if x.weekday() >= 5 else "Weekday"
    )

    rows_per_page = 100
    total_pages = max(1, math.ceil(len(preview_df) / rows_per_page))

    page = st.slider("Page", 1, total_pages, 1)

    start = (page - 1) * rows_per_page
    end = start + rows_per_page

    st.write(f"Showing rows {start} to {min(end, len(preview_df))}")

    st.dataframe(
        preview_df.iloc[start:end],
        use_container_width=True
    )

    st.subheader("📅 Calendar View (Weekday vs Weekend)")

    months = df["timestamp"].dt.to_period("M").unique()

    for m in months:
        year = m.year
        month = m.month

        st.markdown(f"### {calendar.month_name[month]} {year}")

        cal = calendar.monthcalendar(year, month)
        table = []

        for week in cal:
            row = []

            for day in week:
                if day == 0:
                    row.append("")
                else:
                    dt = pd.Timestamp(year=year, month=month, day=day)

                    if dt.weekday() >= 5:
                        row.append(f"🟥 {day}")
                    else:
                        row.append(f"🟩 {day}")

            table.append(row)

        st.table(
            pd.DataFrame(
                table,
                columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            )
        )

    st.info(
        """
🟩 Weekday → Used for Maximum Demand calculation

🟥 Weekend → Ignored in Maximum Demand calculation
"""
    )

if st.button("Run Simulation"):

        results_df = run_bess_matrix(
            df=df,
            max_qty=max_qty,
            unit_kw=unit_kw,
            unit_kwh=unit_kwh,
            dod=dod,
            buffer_pct=buffer_pct,
            capex_per_unit=capex_per_unit,
            savings_per_kw=savings_per_kw,
            enable_discharge=enable_discharge,
        )

        st.subheader("📈 Simulation Results")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("⚡ Peak Shaved (kW)")
        st.bar_chart(
            results_df.set_index("BESS Qty")["Peak Shaved (kW)"]
        )

        st.subheader("💰 Monthly Savings (RM)")
        st.bar_chart(
            results_df.set_index("BESS Qty")["Monthly Savings (RM)"]
        )

        st.subheader("⭐ Recommended Configuration")

        best = best_system(results_df)

        st.success(
            f"Recommended System: {int(best['BESS Qty'])} Container(s)"
        )

        st.dataframe(
            pd.DataFrame([best]),
            use_container_width=True
        )

        st.subheader("🔥 Highest Daily Load per Month (14:00–22:00 window)")

        weekday_monthly = get_monthly_highest_daily_peak(
            df,
            day_type="weekday"
        )

        weekend_monthly = get_monthly_highest_daily_peak(
            df,
            day_type="weekend"
        )

        c_weekday, c_weekend = st.columns(2)

        with c_weekday:
            st.markdown("**Weekdays**")

            if weekday_monthly.empty:
                st.info("No weekday data found in the selected window.")
            else:
                st.dataframe(
                    weekday_monthly.round(
                        {"highest_daily_peak_kw": 1}
                    ),
                    use_container_width=True,
                )

        with c_weekend:
            st.markdown("**Weekends**")

            if weekend_monthly.empty:
                st.info("No weekend data found in the selected window.")
            else:
                st.dataframe(
                    weekend_monthly.round(
                        {"highest_daily_peak_kw": 1}
                    ),
                    use_container_width=True,
                )
