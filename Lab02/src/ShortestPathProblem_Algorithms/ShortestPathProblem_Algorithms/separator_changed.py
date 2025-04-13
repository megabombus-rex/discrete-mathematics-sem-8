input_file = 'Results/BellmanFord/BellmanFordTests.csv'
output_file = 'Results/BellmanFord/BellmanFordTests_clean.csv'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line_num, line in enumerate(infile):
        if line_num == 0:
            # Write header unchanged
            outfile.write(line)
            continue

        # Find positions of commas
        comma_indices = [i for i, c in enumerate(line) if c == ',']

        if len(comma_indices) >= 4:
            # Replace the 3rd comma (index 2) with a dot
            pos = comma_indices[2]
            line = line[:pos] + '.' + line[pos+1:]

        outfile.write(line)