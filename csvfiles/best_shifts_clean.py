import csv

input_file = r"csvfiles\best_shifts.csv"
output_file = r"csvfiles\best_shifts.csv"

# Read the input file and remove duplicates
with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    unique_lines = set(tuple(row) for row in reader)

# Write the unique lines to the output file
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(unique_lines)

print(f"Duplicate lines removed. Cleaned file saved as '{output_file}'.")