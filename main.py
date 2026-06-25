import pandas as pd

# ==================================
# SETTINGS
# ==================================
UNIT_KWH = 261
UNIT_KW = 125
TARGET_CEILING = 60
SAFETY_BUFFER = 0.10
INTERVAL_HOURS = 0.5

# ==================================
# LOAD DATA
# ==================================
df = pd.read_excel("load_data.xlsx", header=None)
df.columns = df.iloc[2]
df = df.iloc[3:].reset_index(drop=True)

df = df[['Date / End Time', 'kW Import']]
df.columns = ['timestamp', 'load_kw']

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['load_kw'] = pd.to_numeric(df['load_kw'])

peak_df = df[
    (df['timestamp'].dt.hour >= 14) &
    (df['timestamp'].dt.hour < 22)
]

# ==================================
# SIMULATION FUNCTION
# ==================================
def simulate_battery(battery_qty):
    battery_kwh = battery_qty * UNIT_KWH
    battery_kw = battery_qty * UNIT_KW
    buffer_energy = battery_kwh * SAFETY_BUFFER
    battery_soc = battery_kwh

    total_shaved_kw = 0
    battery_failed = False

    for _, row in peak_df.iterrows():
        load = row["load_kw"]

        if load > TARGET_CEILING:
            required_kw = load - TARGET_CEILING
            discharge_kw = min(required_kw, battery_kw)
            energy_used = discharge_kw * INTERVAL_HOURS

            if battery_soc - energy_used <= buffer_energy:
                battery_failed = True
                break

            battery_soc -= energy_used
            total_shaved_kw += discharge_kw

    return {
        "battery_qty": battery_qty,
        "status": "FAILED" if battery_failed else "PASSED",
        "remaining_soc": round(battery_soc, 2),
        "total_shaved_kw": round(total_shaved_kw, 2)
    }

# ==================================
# BUILD BESS MATRIX
# ==================================
results = []

for qty in range(1, 6):
    result = simulate_battery(qty)
    results.append(result)

results_df = pd.DataFrame(results)

print("\n===== BESS MATRIX =====")
print(results_df)