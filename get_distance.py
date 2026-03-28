import requests
import json
import time

# Our 15 Mumbai locations
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

names = list(LOCATIONS.keys())
coords = list(LOCATIONS.values())

def get_distance(origin, destination):
    """Get real road distance and duration from OSRM"""
    lat1, lon1 = origin
    lat2, lon2 = destination
    
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "false"}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data["code"] == "Ok":
            route = data["routes"][0]
            distance_km = round(route["distance"] / 1000, 2)  # meters to km
            duration_min = round(route["duration"] / 60, 2)   # seconds to minutes
            return distance_km, duration_min
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def build_matrix():
    n = len(names)
    distance_matrix = {}
    duration_matrix = {}
    
    print(f"Fetching distances for {n} locations ({n*n} pairs)...")
    print("This will take about 2-3 minutes...\n")
    
    for i in range(n):
        distance_matrix[names[i]] = {}
        duration_matrix[names[i]] = {}
        
        for j in range(n):
            if i == j:
                distance_matrix[names[i]][names[j]] = 0.0
                duration_matrix[names[i]][names[j]] = 0.0
            else:
                dist, dur = get_distance(coords[i], coords[j])
                
                if dist is not None:
                    distance_matrix[names[i]][names[j]] = dist
                    duration_matrix[names[i]][names[j]] = dur
                    print(f"✅ {names[i]} → {names[j]}: {dist}km, {dur}min")
                else:
                    # Fallback: straight line estimate
                    distance_matrix[names[i]][names[j]] = -1
                    duration_matrix[names[i]][names[j]] = -1
                    print(f"❌ Failed: {names[i]} → {names[j]}")
                
                time.sleep(0.5)  # Be nice to OSRM server
    
    return distance_matrix, duration_matrix

if __name__ == "__main__":
    dist_matrix, dur_matrix = build_matrix()
    
    result = {
        "locations": names,
        "coordinates": {name: {"lat": coords[i][0], "lon": coords[i][1]} 
                       for i, name in enumerate(names)},
        "distance_km": dist_matrix,
        "duration_min": dur_matrix
    }
    
    with open("distance_matrix.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n✅ Done! Saved to distance_matrix.json")