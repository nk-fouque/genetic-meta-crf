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
    cities = read_words('preprocessing_data/cities')

    with open(input, 'r') as file:
        filedata = file.read()

    # Replace the target string
    for state in states:
        filedata = filedata.replace(state+'\t', 'state-token\t')
    for city in cities:
        filedata = filedata.replace(city+'\t', 'city-token\t')
        filedata = filedata.replace(city+'-airport\t', 'city-airport-token\t')

    # Write the file out again
    with open(output, 'w') as file:
        file.write(filedata)
