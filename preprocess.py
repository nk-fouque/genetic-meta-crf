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
    days = read_words('preprocessing_data/days')

    with open(input, 'r') as file:
        filedata = file.read()

    # Replace the target string
    for state in states:
        filedata = filedata.replace(state+'\n', 'state-token\n')
    for city in cities:
        filedata = filedata.replace(city+'\n', 'city-token\n')
        filedata = filedata.replace(city+'-airport\n', 'city-airport-token\n')
    for day in days:
        filedata = filedata.replace(day+'\n', 'day-token\n')

    # Write the file out again
    with open(output, 'w') as file:
        file.write(filedata)
