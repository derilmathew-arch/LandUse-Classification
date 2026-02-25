import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(r"C:\Users\Lenovo\Downloads\AER.jpg")

if image is None:
    print("Error: Image not found. Check your file path.")
    exit()

image = cv2.resize(image, (800, 600))

# Save original for report
cv2.imwrite(os.path.join(output_dir, "original_image.png"), image)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

kernel = np.ones((5, 5), np.uint8)

# -----------------------------
# VEGETATION
# -----------------------------
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# WATER
# -----------------------------
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# ROADS
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

mask_road = cv2.bitwise_and(bright_mask, cv2.bitwise_not(mask_green))
mask_road = cv2.bitwise_and(mask_road, cv2.bitwise_not(mask_blue))
mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel)

# -----------------------------
# BUILDINGS
# -----------------------------
combined = cv2.bitwise_or(mask_green, mask_blue)
combined = cv2.bitwise_or(combined, mask_road)

mask_building = cv2.bitwise_not(combined)
mask_building = cv2.morphologyEx(mask_building, cv2.MORPH_OPEN, kernel)

# -----------------------------
# AREA CALCULATION
# -----------------------------
total_pixels = image.shape[0] * image.shape[1]

green_percent = (cv2.countNonZero(mask_green) / total_pixels) * 100
blue_percent = (cv2.countNonZero(mask_blue) / total_pixels) * 100
road_percent = (cv2.countNonZero(mask_road) / total_pixels) * 100
building_percent = (cv2.countNonZero(mask_building) / total_pixels) * 100

print("\n===== LAND COVER ANALYSIS =====")
print(f"Vegetation: {green_percent:.2f}%")
print(f"Water: {blue_percent:.2f}%")
print(f"Roads: {road_percent:.2f}%")
print(f"Buildings: {building_percent:.2f}%")

# -----------------------------
# DRAW CONTOURS
# -----------------------------
output = image.copy()

for mask, color in [
    (mask_green, (0, 255, 0)),
    (mask_blue, (255, 0, 0)),
    (mask_road, (0, 0, 255)),
    (mask_building, (0, 255, 255))
]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)

# -----------------------------
# SAVE OUTPUT IMAGES
# -----------------------------
cv2.imwrite(os.path.join(output_dir, "vegetation_mask.png"), mask_green)
cv2.imwrite(os.path.join(output_dir, "water_mask.png"), mask_blue)
cv2.imwrite(os.path.join(output_dir, "road_mask.png"), mask_road)
cv2.imwrite(os.path.join(output_dir, "building_mask.png"), mask_building)
cv2.imwrite(os.path.join(output_dir, "segmented_image.png"), output)

# -----------------------------
# SAVE PIE CHART
# -----------------------------
plt.figure()
plt.pie(
    [green_percent, blue_percent, road_percent, building_percent],
    labels=["Vegetation", "Water", "Road", "Building"],
    autopct="%1.1f%%"
)
plt.title("Land Use Distribution")
plt.savefig(os.path.join(output_dir, "pie_chart.png"))
plt.close()

# -----------------------------
# SAVE CSV
# -----------------------------
df = pd.DataFrame([[
    "AER.jpg", green_percent, blue_percent, road_percent, building_percent
]],
columns=["Image", "Vegetation %", "Water %", "Road %", "Building %"])

df.to_csv(os.path.join(output_dir, "result.csv"), index=False)

# -----------------------------
# DISPLAY
# -----------------------------
cv2.imshow("Segmented Image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()