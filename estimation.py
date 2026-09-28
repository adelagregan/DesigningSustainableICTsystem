# Footprint model for our camera-trap system (Sustainable AI assignment)
# Computes the numbers in our Environmental Impact Assessment.
# Run with: python3 footprint_model.py
#
# Sources are given as bib keys from our report. "assumed" means our own estimate.

base = {
    # field unit
    "mode": "gated",          # gated (our design), always_on, or continuous
    "p_idle": 0.6,            # W, Pi Zero 2 W idle [cnxsoftwarepizero2w]
    "p_load": 2.5,            # W, Pi under load [geerlingzero2]
    "p_sleep": 0.005,         # W, microcontroller + PIR (assumed, datasheets give < 1 mW)
    "e_boot": 30,             # J per daily boot (assumed)
    "e_capture": 1.0,         # J per photo (assumed)
    "e_detect": 5.0,          # J per MegaDetector run (assumed: ~2 s at 2.5 W)
    "e_classify": 0.5,        # J per classification (assumed)
    "triggers": 50,           # per day (assumed)
    "animal_share": 0.3,      # share of triggers with an animal (assumed)

    # battery and panel sizing
    "backup_days": 3,         # assumed
    "dod": 0.8,               # usable battery capacity (assumed)
    "min_battery_wh": 5,      # smallest practical battery (assumed)
    "sun_hours": 4.5,         # [malaysiasolar]
    "efficiency": 0.7,        # assumed
    "sunlight": 0.25,         # share of sun in a canopy gap (assumed, 10-50% [canopygaps])
    "min_panel_w": 2,         # smallest practical panel (assumed)

    # manufacturing emissions
    "ef_battery": 95,         # kg CO2e/kWh [lfp2024, ademelfp]
    "ef_panel": 515,          # kg CO2e/kW [gecsolar]
    "core_kg": 8.0,           # board, camera, PIR, housing, SD (TODO: from our component table)
    "mcu_kg": 0.5,            # microcontroller, only in our design (assumed)
    "batteries": 1,           # over 3 years [preger2020]
    "panels": 1.5,            # over 3 years (assumed)

    # training (done once for the whole fleet)
    "gpu_kw": 0.07,           # [nvidiat4]
    "pue": 1.2,               # [li2023thirsty]
    "grid_training": 0.48,    # kg CO2e/kWh [emberglobal2024]
    "configs": 20,            # assumed
    "extract_h": 0.5,         # assumed
    "config_h": 0.05,         # assumed
    "reuse_crops": True,

    # transport
    "ef_car": 0.25,           # kg CO2e/km (assumed, DEFRA average car is 0.16 [defraconversion])
    "km_per_trip": 150,       # assumed
    "units_per_trip": 10,     # assumed
    "trips_per_year": 1,

    "years": 3,
    "fleet": 50,              # assumed

    # water (training only) [li2023thirsty]
    "wue_on": 0.55,
    "wue_off": 3.14,
}


def energy_per_day(p):
    """Energy the unit uses per day, in Wh."""
    photos = p["triggers"] * (p["e_capture"] + p["e_detect"]
                              + p["animal_share"] * p["e_classify"]) / 3600
    if p["mode"] == "gated":
        return p["p_sleep"] * 24 + p["e_boot"] / 3600 + photos
    if p["mode"] == "always_on":
        return p["p_idle"] * 24 + photos
    if p["mode"] == "continuous":
        return p["p_load"] * 24  # ignores camera and night lighting, so a lower bound


def training_kwh(p):
    if p["reuse_crops"]:
        hours = p["extract_h"] + p["configs"] * p["config_h"]
    else:
        hours = p["configs"] * (p["extract_h"] + p["config_h"])
    return p["gpu_kw"] * hours * p["pue"]


def footprint(p):
    """Footprint per unit over the whole deployment, in kg CO2e."""
    e_day = energy_per_day(p)
    battery_wh = max(e_day * p["backup_days"] / p["dod"], p["min_battery_wh"])
    panel_w = max(e_day / (p["sun_hours"] * p["efficiency"] * p["sunlight"]), p["min_panel_w"])

    core = p["core_kg"] + (p["mcu_kg"] if p["mode"] == "gated" else 0)
    battery = p["batteries"] * battery_wh / 1000 * p["ef_battery"]
    panel = p["panels"] * panel_w / 1000 * p["ef_panel"]
    training = training_kwh(p) * p["grid_training"] / p["fleet"]
    transport = p["years"] * p["trips_per_year"] * p["km_per_trip"] * p["ef_car"] / p["units_per_trip"]

    parts = {"core": core, "battery": battery, "panel": panel,
             "training": training, "transport": transport}
    return sum(parts.values()), parts, battery_wh, panel_w


def with_changes(**changes):
    p = dict(base)
    p.update(changes)
    return p


# --- results for our design ---
total, parts, battery_wh, panel_w = footprint(base)
print("Our design")
print(f"  energy: {energy_per_day(base):.2f} Wh/day (always-on Pi would idle at {base['p_idle'] * 24:.1f} Wh/day)")
print(f"  training: {training_kwh(base):.2f} kWh, without reusing crops "
      f"{training_kwh(with_changes(reuse_crops=False)):.2f} kWh")
print(f"  battery {battery_wh:.0f} Wh, panel {panel_w:.0f} W")
for name, value in parts.items():
    print(f"  {name:10s} {value:6.2f} kg ({value / total:.0%})")
print(f"  total {total:.1f} kg per unit, {total * base['fleet']:.0f} kg for the fleet")

e_it = training_kwh(base) / base["pue"]
water = e_it * base["wue_on"] + e_it * base["pue"] * base["wue_off"]
print(f"  training water: {water:.2f} L")

# --- decision table: change one decision at a time ---
print("\nDecision table")
decisions = {
    "Continuous inference, no PIR": with_changes(mode="continuous"),
    "Store all images (4 trips/year)": with_changes(trips_per_year=4),
    "Pi always on": with_changes(mode="always_on"),
    "MegaDetector V5 (>= 300 J/photo)": with_changes(e_detect=300),
    "Replace battery yearly": with_changes(batteries=3),
    "Don't reuse crops in training": with_changes(reuse_crops=False),
}
for name, p in decisions.items():
    t, _, b, pv = footprint(p)
    print(f"  {name:34s} {t:6.1f} kg ({t / total - 1:+.0%}), battery {b:.0f} Wh, panel {pv:.0f} W")

# --- sensitivity: change one assumption at a time ---
print("\nSensitivity (low -- high)")
tests = {
    "Units per trip (20 / 3)": ({"units_per_trip": 20}, {"units_per_trip": 3}),
    "Trip distance (50 / 300 km)": ({"km_per_trip": 50}, {"km_per_trip": 300}),
    "Core hardware (4 / 16 kg)": ({"core_kg": 4}, {"core_kg": 16}),
    "Sunlight (50% / 10%)": ({"sunlight": 0.5}, {"sunlight": 0.1}),
    "Workload (10x2 J / 250x15 J)": ({"triggers": 10, "e_detect": 2},
                                     {"triggers": 250, "e_detect": 15}),
    "Smallest parts (2 Wh+1 W / 10 Wh+5 W)": ({"min_battery_wh": 2, "min_panel_w": 1},
                                             {"min_battery_wh": 10, "min_panel_w": 5}),
}
for name, (low, high) in tests.items():
    print(f"  {name:38s} {footprint(with_changes(**low))[0]:6.1f} -- {footprint(with_changes(**high))[0]:6.1f}")

best = with_changes(units_per_trip=20, km_per_trip=50)
worst = with_changes(units_per_trip=3, km_per_trip=300)
print(f"  {'Transport best / worst combined':38s} {footprint(best)[0]:6.1f} -- {footprint(worst)[0]:6.1f}")
print(f"  {'Sunlight, always-on Pi':38s} "
      f"{footprint(with_changes(mode='always_on', sunlight=0.5))[0]:6.1f} -- "
      f"{footprint(with_changes(mode='always_on', sunlight=0.1))[0]:6.1f}")