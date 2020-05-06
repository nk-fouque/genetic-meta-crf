from typing import List
import sys

def read_words(file: str) -> List[str]:
    open_file = open(file, 'r')
    contents = open_file.read().split()
    open_file.close()
    return contents

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    states = read_words('preprocessing_data/state')

    with open(input, 'r') as file:
        filedata = file.read()

    # Replace the target string
    for state in states:
        filedata[:] = [s.replace(state+'\t', 'state-token\t') for s in filedata]

    # Write the file out again
    with open(output, 'w') as file:
        file.write(filedata)
