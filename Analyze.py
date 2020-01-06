import time
import os

brands = {}

print("[INFO] Initializing file scan")
file_scan_start = time.time()

# r=root, d=directories, f = files
for r, d, f in os.walk("VMMRdb/"):
    for directory in d:
        brand = directory.split('_')
        if brand[0] not in brands:
            brands[brand[0]] = 0

    for file in f:
        directory = os.path.join(r, file).split('/')[-1]
        label = str(directory.split('_')[0])
        brands[label] += 1

print("[INFO] Scan complete")
print("[INFO]      Took " + str(round(time.time() - file_scan_start, 2)) + " s")
print("Brand count: " + str(len(brands)))

sorted_brands = reversed(sorted(brands.items(), key=lambda kv: kv[1]))
# for brand in sorted_brands:
#     print(brand[0] + ": " + str(brand[1]))

file = open("analyze_result.txt", "w")
for brand in sorted_brands:
    file.write(brand[0] + ": " + str(brand[1]) + '\n')
file.truncate()
file.close()
print("Output saved to analyze_result.txt")
