import streamlit as st
import pandas as pd
import calendar
import math
import altair as alt

from simulation import (
    load_and_clean_data,
    run_bess_matrix,
    best_system,
    get_monthly_highest_daily_peak,
    get_weekday_onpeak_max_load,
    get_saved_daily_diagnostics,
    get_saved_degradation,
    get_peak_shaving_profile,
)


st.set_page_config(
    page_title="BESS Proposal Generator",
    layout="wide",
)

st.title("BESS Peak Shaving Proposal Generator")


def format_rm(value):
    if pd.isna(value):
        return "-"

    return f"RM {value:,.0f}"


def hour_label(hour):
    if hour == 0:
        return "12:00 AM"

    if hour < 12:
        return f"{hour}:00 AM"

    if hour == 12:
        return "12:00 PM"

    if hour == 24:
        return "12:00 AM"

    return f"{hour - 12}:00 PM"


def calendar_html(year, month):
    weeks = calendar.monthcalendar(year, month)

    html = """
    <style>
    .cal-wrap {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 28px;
        font-size: 15px;
    }
    .cal-wrap th {
        text-align: left;
        color: #a8b3c7;
        padding: 10px;
        border: 1px solid #263040;
    }
    .cal-wrap td {
        height: 48px;
        padding: 10px;
        border: 1px solid #263040;
        vertical-align: top;
        font-weight: 600;
    }
    .weekday-cell {
        background: rgba(34, 197, 94, 0.18);
        color: #d9f99d;
    }
    .weekend-cell {
        background: rgba(239, 68, 68, 0.18);
        color: #fecaca;
    }
    .empty-cell {
        background: transparent;
    }
    .legend {
        margin-bottom: 10px;
        color: #cbd5e1;
        font-size: 14px;
    }
    .dot-green {
        display: inline-block;
        width: 11px;
        height: 11px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-red {
        display: inline-block;
        width: 11px;
        height: 11px;
        background: #ef4444;
        border-radius: 50%;
        margin-left: 18px;
        margin-right: 6px;
    }
    </style>
    """

    html += """
    <div class="legend">
        <span class="dot-green"></span>Weekday
        <span class="dot-red"></span>Weekend
    </div>
    """

    html += "<table class='cal-wrap'>"
    html += "<tr>"

    for name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        html += f"<th>{name}</th>"

    html += "</tr>"

    for week in weeks:
        html += "<tr>"

        for day in week:
            if day == 0:
                html += "<td class='empty-cell'></td>"
            else:
                dt = pd.Timestamp(year=year, month=month, day=day)
                css_class = "weekend-cell" if dt.weekday() >= 5 else "weekday-cell"
                html += f"<td class='{css_class}'>{day}</td>"

        html += "</tr>"

    html += "</table>"

    return html


def get_window_times(profile_df, start_hour, end_hour):
    chart_date = profile_df["timestamp"].dt.normalize().iloc[0]
    window_start = chart_date + pd.Timedelta(hours=start_hour)
    window_end = chart_date + pd.Timedelta(hours=end_hour)

    return window_start, window_end


def add_window_endpoint(df, window_end):
    if df.empty:
        return df

    if df["timestamp"].max() >= window_end:
        return df

    last_row = df.iloc[-1].copy()
    last_row["timestamp"] = window_end

    if "interval_end" in last_row:
        last_row["interval_end"] = window_end

    return pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)


def build_peak_shaving_chart(profile_df, start_hour, end_hour):
    if profile_df.empty:
        return None

    window_start, window_end = get_window_times(profile_df, start_hour, end_hour)

    chart_df = profile_df[
        [
            "timestamp",
            "load_kw",
            "grid_kw_after_bess",
            "target_kw",
        ]
    ].copy()

    chart_df = chart_df.rename(
        columns={
            "load_kw": "Original Load",
            "grid_kw_after_bess": "Grid After BESS",
            "target_kw": "Target Peak",
        }
    )

    chart_df = chart_df[
        (chart_df["timestamp"] >= window_start)
        & (chart_df["timestamp"] < window_end)
    ].copy()

    if chart_df.empty:
        return None

    interval_minutes = (
        chart_df["timestamp"]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
        .median()
        / 60
    )

    if pd.isna(interval_minutes) or interval_minutes <= 0:
        interval_minutes = 30

    chart_df["interval_end"] = chart_df["timestamp"] + pd.to_timedelta(
        interval_minutes,
        unit="m",
    )

    chart_df.loc[
        chart_df["interval_end"] > window_end,
        "interval_end",
    ] = window_end

    line_df = add_window_endpoint(chart_df.copy(), window_end)

    shaving_df = chart_df[
        (chart_df["Original Load"] > chart_df["Target Peak"])
        & (chart_df["Grid After BESS"] < chart_df["Original Load"])
    ].copy()

    x_scale = alt.Scale(
        domain=[
            window_start.to_pydatetime(),
            window_end.to_pydatetime(),
        ]
    )

    x_axis = alt.Axis(
        format="%I:%M %p",
        tickCount=9,
        labelAngle=0,
        title="Time",
    )

    original_line = (
        alt.Chart(line_df)
        .mark_line(color="#8A94A6", opacity=0.75, strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", scale=x_scale, axis=x_axis),
            y=alt.Y("Original Load:Q", title="Load (kW)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%I:%M %p"),
                alt.Tooltip("Original Load:Q", title="Original Load", format=",.1f"),
            ],
        )
    )

    bess_line = (
        alt.Chart(line_df)
        .mark_line(color="#2DD4BF", strokeWidth=3.5)
        .encode(
            x=alt.X("timestamp:T", scale=x_scale, axis=x_axis),
            y=alt.Y("Grid After BESS:Q", title="Load (kW)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%I:%M %p"),
                alt.Tooltip("Grid After BESS:Q", title="Grid After BESS", format=",.1f"),
            ],
        )
    )

    target_line = (
        alt.Chart(line_df)
        .mark_line(color="#F59E0B", strokeDash=[8, 5], strokeWidth=2.5)
        .encode(
            x=alt.X("timestamp:T", scale=x_scale, axis=x_axis),
            y=alt.Y("Target Peak:Q", title="Load (kW)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("Target Peak:Q", title="Target Peak", format=",.1f"),
            ],
        )
    )

    if shaving_df.empty:
        return (original_line + bess_line + target_line).properties(height=420)

    shaved_area = (
        alt.Chart(shaving_df)
        .mark_rect(opacity=0.20, color="#38BDF8")
        .encode(
            x=alt.X("timestamp:T", scale=x_scale, axis=x_axis),
            x2=alt.X2("interval_end:T"),
            y=alt.Y("Grid After BESS:Q", title="Load (kW)", scale=alt.Scale(zero=True)),
            y2=alt.Y2("Original Load:Q"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%I:%M %p"),
                alt.Tooltip("Original Load:Q", title="Original Load", format=",.1f"),
                alt.Tooltip("Grid After BESS:Q", title="After BESS", format=",.1f"),
                alt.Tooltip("Target Peak:Q", title="Target Peak", format=",.1f"),
            ],
        )
    )

    return (shaved_area + original_line + bess_line + target_line).properties(height=420)


def build_soc_chart(profile_df, start_hour, end_hour):
    if profile_df.empty:
        return None

    window_start, window_end = get_window_times(profile_df, start_hour, end_hour)

    soc_df = profile_df.copy()

    soc_df = soc_df[
        (soc_df["timestamp"] >= window_start)
        & (soc_df["timestamp"] < window_end)
    ].copy()

    if soc_df.empty:
        return None

    soc_df = add_window_endpoint(soc_df, window_end)

    return (
        alt.Chart(soc_df)
        .mark_line(color="#2DD4BF", strokeWidth=3.5)
        .encode(
            x=alt.X(
                "timestamp:T",
                title="Time",
                scale=alt.Scale(
                    domain=[
                        window_start.to_pydatetime(),
                        window_end.to_pydatetime(),
                    ]
                ),
                axis=alt.Axis(format="%I:%M %p", tickCount=9, labelAngle=0),
            ),
            y=alt.Y(
                "soc_pct:Q",
                title="Battery SOC (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%I:%M %p"),
                alt.Tooltip("soc_pct:Q", title="SOC (%)", format=",.1f"),
                alt.Tooltip("action:N", title="Action"),
            ],
        )
        .properties(height=280)
    )


def make_run_params(
    data_key,
    unit_kw,
    unit_kwh,
    dod,
    buffer_pct,
    max_qty,
    start_hour,
    end_hour,
    enable_opp_charging,
    capex_per_unit,
    savings_per_kw,
    degradation_pct,
    project_years,
):
    return {
        "data_key": data_key,
        "unit_kw": float(unit_kw),
        "unit_kwh": float(unit_kwh),
        "dod": float(dod),
        "buffer_pct": float(buffer_pct),
        "max_qty": int(max_qty),
        "start_hour": int(start_hour),
        "end_hour": int(end_hour),
        "enable_opp_charging": bool(enable_opp_charging),
        "capex_per_unit": float(capex_per_unit),
        "savings_per_kw": float(savings_per_kw),
        "degradation_pct": float(degradation_pct),
        "project_years": int(project_years),
    }


uploaded_file = st.file_uploader(
    "Upload Load Profile Excel",
    type=["xlsx"],
)


st.sidebar.header("System Parameters")

unit_kw = st.sidebar.number_input(
    "kW per Container",
    min_value=1.0,
    value=125.0,
    step=1.0,
)

unit_kwh = st.sidebar.number_input(
    "kWh per Container",
    min_value=1.0,
    value=261.0,
    step=1.0,
)

dod = st.sidebar.number_input(
    "Depth of Discharge",
    min_value=0.01,
    max_value=1.00,
    value=0.95,
    step=0.01,
)

buffer_percent = st.sidebar.slider(
    "Safety Buffer (%)",
    min_value=0,
    max_value=80,
    value=10,
    step=1,
)

buffer_pct = buffer_percent / 100

max_qty = st.sidebar.number_input(
    "Maximum Containers",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
)

st.sidebar.header("Peak Shaving Window")

hour_options = list(range(0, 25))

start_hour = st.sidebar.selectbox(
    "Start Hour",
    options=hour_options[:-1],
    index=14,
    format_func=hour_label,
)

end_hour = st.sidebar.selectbox(
    "End Hour",
    options=hour_options[1:],
    index=21,
    format_func=hour_label,
)

if end_hour <= start_hour:
    st.sidebar.error("End hour must be later than start hour.")

st.sidebar.header("Operation Mode")

enable_opp_charging = st.sidebar.toggle(
    "Enable Opportunity Charging",
    value=True,
)

st.sidebar.header("Financial Parameters")

capex_per_unit = st.sidebar.number_input(
    "Capex per Container (RM)",
    min_value=0.0,
    value=200000.0,
    step=1000.0,
)

savings_per_kw = st.sidebar.number_input(
    "Peak Demand Tariff (RM/kW/month)",
    min_value=0.0,
    value=97.06,
    step=1.0,
)

degradation_pct = st.sidebar.number_input(
    "Annual Degradation (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
)

project_years = st.sidebar.number_input(
    "Financial Projection Years",
    min_value=1,
    max_value=30,
    value=10,
    step=1,
)


if uploaded_file:
    try:
        df = load_and_clean_data(uploaded_file)

        st.success("Load profile imported successfully")

        detected_time_col = df.attrs.get("detected_time_column", "timestamp")
        detected_load_col = df.attrs.get("detected_load_column", "load_kw")

        st.caption(
            f"Detected timestamp column: {detected_time_col} | Detected load column: {detected_load_col}"
        )

        total_rows = len(df)
        start_date = df["timestamp"].min()
        end_date = df["timestamp"].max()

        data_key = (
            uploaded_file.name,
            total_rows,
            str(start_date),
            str(end_date),
            float(df["load_kw"].max()),
        )

        weekday_df = df[df["timestamp"].dt.weekday < 5]
        weekend_df = df[df["timestamp"].dt.weekday >= 5]

        weekday_onpeak_max = get_weekday_onpeak_max_load(
            df,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        st.subheader("Load Profile Summary")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Total Rows", f"{total_rows:,}")
        c2.metric("Weekdays", weekday_df["timestamp"].dt.date.nunique())
        c3.metric("Weekends", weekend_df["timestamp"].dt.date.nunique())
        c4.metric("Months", len(df["timestamp"].dt.to_period("M").unique()))
        c5.metric("Highest Weekday On-Peak", f"{weekday_onpeak_max:,.1f} kW")

        c6, c7, c8 = st.columns(3)

        c6.metric("Start Date", start_date.strftime("%Y-%m-%d %I:%M %p"))
        c7.metric("End Date", end_date.strftime("%Y-%m-%d %I:%M %p"))
        c8.metric("On-Peak Window", f"{hour_label(start_hour)} - {hour_label(end_hour)}")

        with st.expander("Full Data Preview", expanded=False):
            rows_per_page = 100
            total_pages = max(1, math.ceil(len(df) / rows_per_page))

            page = st.slider("Page", 1, total_pages, 1)

            start = (page - 1) * rows_per_page
            end = start + rows_per_page

            preview = df.iloc[start:end].copy()
            preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%d %I:%M %p")

            st.dataframe(preview, use_container_width=True)

        with st.expander("Calendar View", expanded=False):
            months = df["timestamp"].dt.to_period("M").unique()

            for m in months:
                st.markdown(f"### {calendar.month_name[m.month]} {m.year}")
                st.markdown(calendar_html(m.year, m.month), unsafe_allow_html=True)

        current_params = make_run_params(
            data_key=data_key,
            unit_kw=unit_kw,
            unit_kwh=unit_kwh,
            dod=dod,
            buffer_pct=buffer_pct,
            max_qty=max_qty,
            start_hour=start_hour,
            end_hour=end_hour,
            enable_opp_charging=enable_opp_charging,
            capex_per_unit=capex_per_unit,
            savings_per_kw=savings_per_kw,
            degradation_pct=degradation_pct,
            project_years=project_years,
        )

        if (
            "results_df" in st.session_state
            and "run_params" in st.session_state
            and st.session_state["run_params"] != current_params
        ):
            st.warning("Settings or uploaded file changed. Click Run Simulation again to refresh the result.")

        if st.button("Run Simulation", type="primary", disabled=end_hour <= start_hour):
            results_df = run_bess_matrix(
                df=df,
                max_qty=max_qty,
                unit_kw=unit_kw,
                unit_kwh=unit_kwh,
                dod=dod,
                buffer_pct=buffer_pct,
                capex_per_unit=capex_per_unit,
                savings_per_kw=savings_per_kw,
                enable_opp_charging=enable_opp_charging,
                degradation_pct=degradation_pct,
                project_years=project_years,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            st.session_state["results_df"] = results_df
            st.session_state["run_params"] = current_params

        if (
            "results_df" in st.session_state
            and "run_params" in st.session_state
            and st.session_state["run_params"] == current_params
        ):
            results_df = st.session_state["results_df"]
            best = best_system(results_df)

            st.subheader("Recommended System")

            r1, r2, r3, r4, r5 = st.columns(5)

            r1.metric("Recommended Qty", f"{int(best['BESS Qty'])} Units")
            r2.metric("Target Peak", f"{best['Target Peak (kW)']:,.1f} kW")
            r3.metric("Peak Shaved", f"{best['Peak Shaved (kW)']:,.1f} kW")
            r4.metric("Annual Savings", format_rm(best["Annual Savings (RM)"]))
            r5.metric("ROI", f"{best['ROI (Years)']:,.2f} Years")

            st.subheader("BESS Decision Matrix")

            matrix_view = results_df[
                [
                    "BESS Qty",
                    "Usable Energy (kWh)",
                    "Peak Shaved (kW)",
                    "Target Peak (kW)",
                    "Buffer (%)",
                    "Capex (RM)",
                    "Monthly Savings (RM)",
                    "Annual Savings (RM)",
                    "ROI (Years)",
                    "Hardest Day",
                    "Lowest SOC (%)",
                    "Energy Charged (kWh)",
                    "Opportunity Charging",
                    "Recommendation",
                ]
            ].copy()

            matrix_view["Hardest Day"] = matrix_view["Hardest Day"].astype(str)

            st.dataframe(matrix_view, use_container_width=True, hide_index=True)

            st.subheader("Client Proposal Infographic")

            qty_list = results_df["BESS Qty"].astype(int).tolist()
            best_qty = int(best["BESS Qty"])

            selected_qty = st.selectbox(
                "Select BESS Scenario",
                options=qty_list,
                index=qty_list.index(best_qty),
                format_func=lambda x: f"{x} Unit{'s' if x > 1 else ''}",
            )

            selected_row = results_df[results_df["BESS Qty"] == selected_qty].iloc[0]

            p1, p2, p3, p4 = st.columns(4)

            p1.metric("BESS Qty", f"{int(selected_row['BESS Qty'])} Units")
            p2.metric("Target Peak", f"{selected_row['Target Peak (kW)']:,.1f} kW")
            p3.metric("Buffer", f"{selected_row['Buffer (%)']:,.1f}%")
            p4.metric("Hardest Day", str(selected_row["Hardest Day"]))

            profile_df = get_peak_shaving_profile(results_df, selected_qty)

            peak_chart = build_peak_shaving_chart(
                profile_df,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            chart_col, info_col = st.columns([4.6, 1.2])

            with chart_col:
                if peak_chart is not None:
                    st.altair_chart(peak_chart, use_container_width=True)
                else:
                    st.warning(
                        "No chart data available for this scenario. Try another BESS quantity or check whether the hardest day has data inside the selected peak window."
                    )

            with info_col:
                st.markdown("### Graph Guide")
                st.markdown(
                    """
                    <div style="line-height:1.9; font-size:15px;">
                        <div><span style="color:#8A94A6;font-weight:700;">━</span> Original Load</div>
                        <div><span style="color:#2DD4BF;font-weight:700;">━</span> Grid After BESS</div>
                        <div><span style="color:#F59E0B;font-weight:700;">- -</span> Target Peak</div>
                        <div><span style="color:#38BDF8;font-weight:700;">■</span> kW Shaved</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.divider()

                st.metric("Peak Shaved", f"{selected_row['Peak Shaved (kW)']:,.1f} kW")
                st.metric("Energy Used", f"{selected_row['Energy Used (kWh)']:,.1f} kWh")
                st.metric("Lowest SOC", f"{selected_row['Lowest SOC (%)']:,.1f}%")

                st.caption(
                    f"Chart shows {hour_label(start_hour)} - {hour_label(end_hour)} for the hardest working day."
                )

            st.subheader("Battery SOC and Opportunity Charging")

            soc_chart = build_soc_chart(
                profile_df,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            if soc_chart is not None:
                st.altair_chart(soc_chart, use_container_width=True)
            else:
                st.info("No SOC chart data available for the selected peak window.")

            o1, o2, o3, o4 = st.columns(4)

            o1.metric("Opportunity Charging", selected_row["Opportunity Charging"])
            o2.metric("Energy Discharged", f"{selected_row['Energy Used (kWh)']:,.1f} kWh")
            o3.metric("Energy Charged", f"{selected_row['Energy Charged (kWh)']:,.1f} kWh")
            o4.metric("Lowest SOC", f"{selected_row['Lowest SOC (%)']:,.1f}%")

            st.subheader("Financial Breakdown")

            f1, f2, f3, f4 = st.columns(4)

            f1.metric("Capex", format_rm(selected_row["Capex (RM)"]))
            f2.metric("Monthly Savings", format_rm(selected_row["Monthly Savings (RM)"]))
            f3.metric("Annual Savings", format_rm(selected_row["Annual Savings (RM)"]))
            f4.metric("ROI", f"{selected_row['ROI (Years)']:,.2f} Years")

            degradation_df = get_saved_degradation(results_df, selected_qty)

            if not degradation_df.empty:
                st.markdown("### Degradation Projection")
                st.dataframe(degradation_df, use_container_width=True, hide_index=True)

                degradation_chart = (
                    alt.Chart(degradation_df)
                    .mark_line(point=True, color="#38bdf8")
                    .encode(
                        x=alt.X("Year:O", title="Year"),
                        y=alt.Y("Cumulative Savings (RM):Q", title="Cumulative Savings (RM)"),
                        tooltip=[
                            alt.Tooltip("Year:O"),
                            alt.Tooltip("Annual Savings (RM):Q", format=",.0f"),
                            alt.Tooltip("Cumulative Savings (RM):Q", format=",.0f"),
                            alt.Tooltip("Net Position (RM):Q", format=",.0f"),
                        ],
                    )
                    .properties(height=300)
                )

                st.altair_chart(degradation_chart, use_container_width=True)

            st.subheader("Hardest Working Day")

            daily = get_saved_daily_diagnostics(results_df, selected_qty)

            if not daily.empty:
                st.dataframe(daily, use_container_width=True, hide_index=True)

            st.subheader("Monthly Peak Demand")

            weekday_monthly = get_monthly_highest_daily_peak(
                df,
                "weekday",
                start_hour=start_hour,
                end_hour=end_hour,
            )

            weekend_monthly = get_monthly_highest_daily_peak(
                df,
                "weekend",
                start_hour=start_hour,
                end_hour=end_hour,
            )

            m1, m2 = st.columns(2)

            with m1:
                st.markdown("Weekdays")
                st.dataframe(weekday_monthly, use_container_width=True)

            with m2:
                st.markdown("Weekends")
                st.dataframe(weekend_monthly, use_container_width=True)

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload an Excel load profile to begin.")
