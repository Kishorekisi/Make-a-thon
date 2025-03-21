# Define candidate rows (Manually calibrated for your table)
candidates = [
    {"name": "CHANDRAN.S", "y_range": (100, 200)},
    {"name": "GOPAL.G.K", "y_range": (200, 300)},
    {"name": "SUMATHI.R", "y_range": (300, 400)},
    {"name": "VENKAT.K", "y_range": (400, 500)}
]

# Find the candidate based on y-coordinate
voted_candidate = None
for candidate in candidates:
    if candidate["y_range"][0] <= seal_y <= candidate["y_range"][1]:
        voted_candidate = candidate["name"]
        break

print(f"Vote detected for: {voted_candidate}")
