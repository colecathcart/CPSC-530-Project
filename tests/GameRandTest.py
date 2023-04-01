import os
import math
import random
import hashlib
from itertools import product
#from scipy.special import comb


def read_moves_from_file(file_path):
    with open(file_path, 'r') as file:
        moves_string = file.read().strip()

    moves = [(int(moves_string[i]), int(moves_string[i + 1])) for i in range(0, len(moves_string), 2)]
    return moves


def calculate_entropy(moves):
    move_counts = {}
    total_moves = len(moves)

    for move in moves:
        move_counts[move] = move_counts.get(move, 0) + 1

    entropy = 0
    for count in move_counts.values():
        probability = count / total_moves
        entropy += -probability * math.log2(probability)

    return entropy


def deterministic_rng(seed, length):
    random.seed(seed)
    numbers = [random.randint(0, 1) for _ in range(length)]
    return numbers


def divide_binary_sequence(sequence, block_length):
    blocks = [sequence[i:i + block_length] for i in range(0, len(sequence), block_length)]
    return blocks


def frequency_test(block):
    n = len(block)
    sum_of_bits = sum(block)
    mean = n / 2
    test_statistic = (sum_of_bits - mean) / math.sqrt(n)
    p_value = math.erfc(abs(test_statistic) / math.sqrt(2))
    return p_value


def runs_test(block):
    n = len(block)
    pi = sum(block) / n
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0

    runs = 1
    for i in range(1, n):
        if block[i] != block[i - 1]:
            runs += 1

    test_statistic = (runs - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))
    p_value = math.erfc(abs(test_statistic) / math.sqrt(2))
    return p_value

def serial_test(block):
    m = 2
    n = len(block)
    counts = {pattern: 0 for pattern in product([0, 1], repeat=m)}

    for i in range(n - m + 1):
        pattern = tuple(block[i:i + m])
        counts[pattern] += 1

    sum_chi_squared = 0
    expected_count = (n - m + 1) / 2 ** m

    for count in counts.values():
        sum_chi_squared += (count - expected_count) ** 2 / expected_count

    p_value = math.erfc(sum_chi_squared / (2 * math.sqrt(2 * (n - m + 1))))
    return p_value

def analyze_nist_results(results, significance=0.15):
    passed_tests = 0
    total_tests = len(results)
    for result in results:
        if result > significance:
            passed_tests += 1
    return (passed_tests / total_tests) * 100

def extractor(moveset):
    decimal_string = ""
    for move in moveset:
        decimal_string += str(move[0])
        decimal_string += str(move[1])
    binary_string = bin(int(decimal_string))[2:]
    sha1string = hashlib.sha1(binary_string.encode())
    return bin((int(sha1string.hexdigest(),16)))[2:130]
    

    
def main():
    file_path = 'moves (2).txt'
    moves = read_moves_from_file(file_path)
    print(moves)
    entropy = calculate_entropy(moves)
    print(entropy)
    normalised_sequence = extractor(moves)
    binary_sequence = deterministic_rng(entropy, 100)
    block_length = 8
    blocks = divide_binary_sequence(binary_sequence, block_length)

    nist_tests = [frequency_test, runs_test, serial_test]
    test_results = []

    for block in blocks:
        block_results = []
        for test in nist_tests:
            p_value = test(block)
            block_results.append(p_value)
        test_results.append(min(block_results))

    percentage_passed = analyze_nist_results(test_results)
    print(f'Percentage of NIST tests passed: {percentage_passed}%')

if __name__ == '__main__':
    main()







