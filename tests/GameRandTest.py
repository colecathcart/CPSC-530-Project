import os
import math
import hashlib
from itertools import product
from scipy.special import comb


def read_moves_from_file(file_path):
    with open(file_path, 'r') as file:
        moves_string = file.read().strip()

    moves = [(int(moves_string[i]), int(moves_string[i + 1])) for i in range(0, len(moves_string), 2)]
    return moves


def calculate_entropy(moves):
    move_counts = {}
    for move in moves:
        move_counts[move] = move_counts.get(move, 0) + 1

    entropy = 0
    remaining_moves = 100

    for count in move_counts.values():
        probability = count / remaining_moves
        entropy += -probability * math.log2(probability)
        remaining_moves -= 1

    return entropy

def convert_moves_to_binary(moves):
    binary_string = ""
    for move in moves:
        binary_string += f"{move[0]:04b}"
        binary_string += f"{move[1]:04b}"
    return [int(bit) for bit in binary_string]


def divide_binary_sequence(sequence, block_length):
    blocks = [sequence[i:i + block_length] for i in range(0, len(sequence), block_length)]
    return blocks

def extractor(moves):
    decimal_string = ""
    for move in moves:
        decimal_string += str(move[0])
        decimal_string += str(move[1])
    binary_string = bin(int(decimal_string))[2:]
    sha1string = hashlib.sha1(binary_string.encode())
    binary_sha1 = bin((int(sha1string.hexdigest(), 16)))[2:130]

    # Convert the binary string to a list of bits
    bit_list = [int(bit) for bit in binary_sha1]
    return bit_list


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

    if pi == 0 or pi == 1:
        return 0.0

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

def analyze_nist_results(results, test_names, significance=0.01):
    passed_tests = 0
    total_tests = len(results)
    for i, result in enumerate(results):
        if result > significance:
            print(f"Test {test_names[i]} passed")
            passed_tests += 1
        else:
            print(f"Test {test_names[i]} failed")
    return (passed_tests / total_tests) * 100

def main():
    file_path = 'moves (6).txt'
    moves = read_moves_from_file(file_path)
    print(moves)

    entropy = calculate_entropy(moves)
    print(entropy)

    block_length = 8

    non_hashed_seq = convert_moves_to_binary(moves)
    print(non_hashed_seq)
    non_hashed_block = divide_binary_sequence(non_hashed_seq, block_length)
    print(non_hashed_block)

    binary_sequence = extractor(moves)
    print(binary_sequence)
    blocks = divide_binary_sequence(binary_sequence, block_length)
    print(blocks)

    nist_tests = [frequency_test, runs_test, serial_test]
    test_names = ["Frequency Test", "Runs Test", "Serial Test"]

    total_tests = 0
    passed_tests = 0

    un_hashed_total_tests = 0
    un_hashed_passed_tests = 0

    for i, block in enumerate(blocks):
        print(f"Block {i + 1}:")
        for j, test in enumerate(nist_tests):
            total_tests += 1
            p_value = test(block)
            if p_value > 0.01:
                print(f"  {test_names[j]}: passed")
                passed_tests += 1
            else:
                print(f"  {test_names[j]}: failed")

    for i, block in enumerate(non_hashed_block):
        print(f"Un-Hashed Block {i + 1}:")
        for j, test in enumerate(nist_tests):
            un_hashed_total_tests += 1
            p_value = test(block)
            if p_value > 0.01:
                print(f"  {test_names[j]}: passed")
                un_hashed_passed_tests += 1
            else:
                print(f"  {test_names[j]}: failed")

    percentage_passed = (passed_tests / total_tests) * 100
    print(f'Percentage of NIST tests passed: {percentage_passed}%')

    non_hashed_percentage_passed = (un_hashed_passed_tests / un_hashed_total_tests) * 100
    print(f'Percentage of NIST tests passed (Un-Hashed): {non_hashed_percentage_passed}%')


if __name__ == '__main__':
    main()






