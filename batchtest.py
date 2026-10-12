import csv
import glob
import subprocess
import sys
 
from algorithms import methods
 
# Output labels from search.py
LABELS = {
    'time_ms': 'Time taken for test',
    'goal': 'Goal reached',
    'nodes_created': 'Nodes created',
    'moves': 'Number of moves',
    'total_cost': 'Total cost',
}
 
# Column header for the CSV
COLUMNS = ['method', *LABELS, 'route']
 
# Writes information to CSV file, one group per test case
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
 
    for test_case in sorted(glob.glob('test_cases/*.txt')): # Gets all test cases and all methods
        writer.writerow([f'Test case: {test_case}']) # Prints the test case being tested above its column headers
        writer.writerow(COLUMNS)
 
        for method in methods:
            output = subprocess.run([sys.executable, 'search.py', test_case, method], capture_output=True, text=True).stdout # Automates test run for all cases and methods
            lines = output.splitlines() # Turns the output into lines
 
            # Extracts information from the output and splits it into 'label' and 'values'
            values = dict(line.split(': ', 1) for line in lines if ': ' in line)
 
            # Builds CSV row with information
            row = {'method': method}
            for column, label in LABELS.items():
                row[column] = values.get(label, '')
 
            # Cleans up the outputs
            row['time_ms'] = row['time_ms'].replace(' ms', '')
            row['goal'] = row['goal'].replace('Node ', '').split(' ')[0]
            row['route'] = lines[-1] if 'Final route:' in lines else ''
 
            writer.writerow([row[column] for column in COLUMNS]) # Writes the row in the same order as the headers
 
        writer.writerow([]) # Blank line to separate current test case from the next
 
print('Saved results to results.csv')