import json

# Load existing distance matrix
with open("distance_matrix.json", "r") as f:
    data = json.load(f)

DISTANCE_KM = data["distance_km"]
LOCATIONS = data["locations"]

# Mumbai signal density
# Main roads: ~1 signal per 0.7km
# Inner roads: ~1 signal per 1.0km
# Average: ~1 signal per 0.8km

SIGNAL_DENSITY = 0.8  # km per signal

# Known heavy signal routes in Mumbai (manual override)
HEAVY_ROUTES = {
    ("Andheri_Station", "Mahakali_Caves"),
    ("Andheri_Station", "Andheri_East"),
    ("NESCO_Warehouse", "Andheri_Station"),
    ("Versova_Beach", "Andheri_Station"),
    ("Andheri_East", "Powai_Lake"),
    ("JVLR_Junction", "Powai_Lake"),
    ("Lokhandwala_Market", "Andheri_Station"),
    ("Goregaon_East", "NESCO_Warehouse"),
}

# Known light signal routes (highways/flyovers)
LIGHT_ROUTES = {
    ("NESCO_Warehouse", "Goregaon_East"),
    ("NESCO_Warehouse", "Bangur_Nagar"),
    ("Bangur_Nagar", "Inorbit_Mall_Malad"),
    ("Inorbit_Mall_Malad", "Bangur_Nagar"),
    ("Powai_Lake", "IIT_Bombay"),
    ("IIT_Bombay", "Powai_Lake"),
    ("Andheri_Station", "Jogeshwari_West"),
    ("Jogeshwari_West", "Andheri_Station"),
}


def calculate_signals(origin, destination):
    dist = DISTANCE_KM[origin][destination]
    base_signals = max(1, round(dist / SIGNAL_DENSITY))

    pair = (origin, destination)
    reverse = (destination, origin)

    if pair in HEAVY_ROUTES or reverse in HEAVY_ROUTES:
        # Add 2-3 extra signals for heavy routes
        return base_signals + 3
    elif pair in LIGHT_ROUTES or reverse in LIGHT_ROUTES:
        # Reduce signals for highway/flyover routes
        return max(1, base_signals - 2)
    else:
        return base_signals


def build_signal_matrix():
    signal_matrix = {}

    for origin in LOCATIONS:
        signal_matrix[origin] = {}
        for dest in LOCATIONS:
            if origin == dest:
                signal_matrix[origin][dest] = 0
            else:
                count = calculate_signals(origin, dest)
                signal_matrix[origin][dest] = count
                print(f"✅ {origin} → {dest}: {count} signals")

    return signal_matrix


if __name__ == "__main__":
    matrix = build_signal_matrix()

    with open("signal_matrix.json", "w") as f:
        json.dump({
            "source": "Distance-based calculation + Mumbai road knowledge",
            "description": (
                "Signal counts derived from real road distances "
                "(OSRM) with Mumbai-specific adjustments for "
                "known heavy/light signal corridors"
            ),
            "signals": matrix
        }, f, indent=2)

    print("\n✅ Done! Saved to signal_matrix.json")