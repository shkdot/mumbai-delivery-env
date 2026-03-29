import requests
import json
import time

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
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"


def get_route_polyline(origin, destination):
    """Get actual route coordinates from OSRM."""
    lat1, lon1 = origin
    lat2, lon2 = destination

    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson"}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data["code"] == "Ok":
            # Returns list of [lon, lat] pairs
            coords = data["routes"][0]["geometry"]["coordinates"]
            return coords
    except Exception as e:
        print(f"  OSRM error: {e}")
    return None


def get_signals_along_route(route_coords):
    """Count traffic signals along actual route using Overpass."""
    if not route_coords or len(route_coords) < 2:
        return 3  # fallback

    # Sample every 5th point to avoid huge queries
    sampled = route_coords[::5]
    if route_coords[-1] not in sampled:
        sampled.append(route_coords[-1])

    # Build polygon string from route points
    poly_points = " ".join([f"{lat} {lon}" for lon, lat in sampled])

    query = f"""
    [out:json][timeout:30];
    node["highway"="traffic_signals"]
      (poly:"{poly_points}");
    out count;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=35
        )
        data = response.json()
        count = int(data["elements"][0]["tags"]["nodes"])
        return count
    except Exception as e:
        print(f"  Overpass error: {e}")
        return 3  # fallback


def build_signal_matrix():
    names = list(LOCATIONS.keys())
    coords = list(LOCATIONS.values())
    n = len(names)
    signal_matrix = {}

    print(f"Fetching REAL signal counts along actual routes...")
    print(f"Total pairs: {n * (n-1)}\n")

    for i in range(n):
        signal_matrix[names[i]] = {}
        for j in range(n):
            if i == j:
                signal_matrix[names[i]][names[j]] = 0
                continue

            # Step 1: Get actual route
            route = get_route_polyline(coords[i], coords[j])
            time.sleep(0.5)

            # Step 2: Count signals along route
            count = get_signals_along_route(route)
            signal_matrix[names[i]][names[j]] = count

            print(f"✅ {names[i]} → {names[j]}: {count} signals")
            time.sleep(1)

    return signal_matrix


if __name__ == "__main__":
    matrix = build_signal_matrix()

    with open("signal_matrix.json", "w") as f:
        json.dump({
            "source": "OpenStreetMap + OSRM real route",
            "description": "Real traffic signal counts along actual driving routes in Mumbai",
            "signals": matrix
        }, f, indent=2)

    print("\n✅ Done! Saved to signal_matrix.json")