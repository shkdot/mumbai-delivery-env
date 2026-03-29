import requests
import json
import time

# Our 15 Mumbai locations with coordinates
LOCATIONS = {
    "NESCO_Warehouse":      (19.1530, 72.8508),
    "Inorbit_Mall_Malad":   (19.1703, 72.8353),
    "Lokhandwala_Market":   (19.1367, 72.8269),
    "Versova_Beach":        (19.1351, 72.8134),
    "Andheri_Station":      (19.1208, 72.8481),
    "Jogeshwari_West":      (19.1200, 72.8500),
    "Reliance_Digital":     (19.1270, 72.8550),
    "Andheri_East":         (19.1136, 72.8697),
    "Mahakali_Caves":       (19.1150, 72.8750),
    "JVLR_Junction":        (19.1050, 72.8800),
    "Powai_Lake":           (19.1176, 72.9060),
    "IIT_Bombay":           (19.1334, 72.9133),
    "Goregaon_East":        (19.1600, 72.8600),
    "Bangur_Nagar":         (19.1650, 72.8350),
    "Decathlon_Andheri":    (19.1180, 72.8320),
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def get_signals_between(origin_coords, dest_coords):
    """
    Count real traffic signals in the bounding box between two locations.
    Uses OpenStreetMap Overpass API.
    """
    lat1, lon1 = origin_coords
    lat2, lon2 = dest_coords

    # Build bounding box with small padding
    min_lat = min(lat1, lat2) - 0.003
    max_lat = max(lat1, lat2) + 0.003
    min_lon = min(lon1, lon2) - 0.003
    max_lon = max(lon1, lon2) + 0.003

    query = f"""
    [out:json][timeout:25];
    node["highway"="traffic_signals"]
      ({min_lat},{min_lon},{max_lat},{max_lon});
    out count;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=30
        )
        data = response.json()
        count = data["elements"][0]["tags"]["nodes"]
        return int(count)
    except Exception as e:
        print(f"  Error: {e}")
        return 3  # default fallback


def build_signal_matrix():
    names = list(LOCATIONS.keys())
    coords = list(LOCATIONS.values())
    n = len(names)

    signal_matrix = {}

    print(f"Fetching real traffic signal counts from OpenStreetMap...")
    print(f"Total pairs: {n * n}\n")

    for i in range(n):
        signal_matrix[names[i]] = {}
        for j in range(n):
            if i == j:
                signal_matrix[names[i]][names[j]] = 0
            else:
                count = get_signals_between(coords[i], coords[j])
                signal_matrix[names[i]][names[j]] = count
                print(f"✅ {names[i]} → {names[j]}: {count} signals")
                time.sleep(1)  # Be nice to Overpass server

    return signal_matrix


if __name__ == "__main__":
    matrix = build_signal_matrix()

    with open("signal_matrix.json", "w") as f:
        json.dump({
            "source": "OpenStreetMap Overpass API",
            "description": "Real traffic signal counts between Mumbai locations",
            "signals": matrix
        }, f, indent=2)

    print("\n✅ Done! Saved to signal_matrix.json")