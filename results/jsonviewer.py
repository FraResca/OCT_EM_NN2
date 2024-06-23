import json
import sys

def print_json_file(filename):
    with open(f'{filename}.json', 'r') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        print(f"Example{i+1}:")
        true = [round(num, 1) for num in item['True']]
        print(f"\tTrue:\t\t{true}")
        predicted = [round(num, 1) for num in item['Predicted']]  # round to 2 decimal places
        print(f"\tPredicted:\t{predicted}")
        print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python jsonviewer.py <filename>")
        sys.exit(1)

    print_json_file(sys.argv[1])