csv_file = "courses.csv"

with open(csv_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Count columns in header
header_cols = len(lines[0].strip().split(","))

# Check each line
for i, line in enumerate(lines[1:], start=2):  # start=2 for line number
    cols = len(line.strip().split(","))
    if cols != header_cols:
        print(f"Line {i} has {cols} columns (expected {header_cols})")
