import os
import math
import hashlib
import copy
import numpy as np
from itertools import product
import scipy.special as spc
import scipy.stats as sst


def read_moves_from_file(file_path):
    with open(file_path, 'r') as file:
        moves_string = file.read().strip()

    moves = [(int(moves_string[i]), int(moves_string[i + 1])) for i in range(0, len(moves_string), 2)]
    return moves


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

#Reference Code: https://gist.github.com/StuartGordonReid/349af3d891e5832272ab
def random_excursion_test(block):
    int_data = np.zeros(len(block))
    for i in range(len(block)):
        if block[i] == '0':
            int_data[i] = -1.0
        else:
            int_data[i] = 1.0

    cumulative_sum = np.cumsum(int_data)
    cumulative_sum = np.append(cumulative_sum, [0])
    cumulative_sum = np.append([0], cumulative_sum)

    x_values = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    position = np.where(cumulative_sum == 0)[0]

    cycles = []
    for pos in range(len(position) - 1):
        cycles.append(cumulative_sum[position[pos]:position[pos + 1] + 1])
    num_cycles = len(cycles)

    state_count = []
    for cycle in cycles:
        state_count.append(([len(np.where(cycle == state)[0]) for state in x_values]))
    state_count = np.transpose(np.clip(state_count, 0, 5))

    su = []
    for cycle in range(6):
        su.append([(sct == cycle).sum() for sct in state_count])
    su = np.transpose(su)

    piks = ([([get_pik_value(uu, state) for uu in range(6)]) for state in x_values])
    inner_term = num_cycles * np.array(piks)
    chi = np.sum(1.0 * (np.array(su) - inner_term) ** 2 / inner_term, axis=1)
    p_values = ([spc.gammaincc(2.5, cs / 2.0) for cs in chi])
    return max(p_values)

def get_pik_value(k, x):
    if k == 0:
        out = 1 - 1.0 / (2 * np.abs(x))
    elif k >= 5:
        out = (1.0 / (2 * np.abs(x))) * (1 - 1.0 / (2 * np.abs(x))) ** 4
    else:
        out = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * np.abs(x))) ** (k - 1)
    return out

#Reference Code: https://gist.github.com/StuartGordonReid/e2d036d9d90ac67f73c0
def random_excursion_variant_test(block):
    int_data = np.zeros(len(block))
    for i in range(len(block)):
        int_data[i] = int(block[i])
    sum_int = (2 * int_data) - np.ones(len(int_data))
    cumulative_sum = np.cumsum(sum_int)

    li_data = []
    for xs in sorted(set(cumulative_sum)):
        if np.abs(xs) <= 9:
            li_data.append([xs, len(np.where(cumulative_sum == xs)[0])])

    j = get_frequency(li_data, 0) + 1
    p_values = []
    for xs in range(-9, 9 + 1):
        if not xs == 0:
            den = np.sqrt(2 * j * (4 * np.abs(xs) - 2))
            p_values.append(spc.erfc(np.abs(get_frequency(li_data, xs) - j) / den))
    max_p_value = max(p_values)
    return max_p_value

def get_frequency(list_data, trigger):
    frequency = 0
    for (x, y) in list_data:
        if x == trigger:
            frequency = y
    return frequency

#Reference code: https://gist.github.com/StuartGordonReid/b9024c910e96d6649a88
def cumulative_sums_test(block, mode='backward'):
    n = len(block)
    counts = np.zeros(n)

    # Calculate the statistic using a walk forward or backwards
    if mode == "backward":
        bin_data = block[::-1]

    ix = 0
    for char in block:
        sub = 1 if char == '1' else -1
        if ix > 0:
            counts[ix] = counts[ix - 1] + sub
        else:
            counts[ix] = sub
        ix += 1

    # This is the maximum absolute level obtained by the sequence
    abs_max = np.max(np.abs(counts))

    start = int(np.floor(0.25 * np.floor(-n / abs_max) + 1))
    end = int(np.floor(0.25 * np.floor(n / abs_max) - 1))
    terms_one = []
    for k in range(start, end + 1):
        sub = sst.norm.cdf((4 * k - 1) * abs_max / np.sqrt(n))
        terms_one.append(sst.norm.cdf((4 * k + 1) * abs_max / np.sqrt(n)) - sub)

    start = int(np.floor(0.25 * np.floor(-n / abs_max - 3)))
    end = int(np.floor(0.25 * np.floor(n / abs_max) - 1))
    terms_two = []
    for k in range(start, end + 1):
        sub = sst.norm.cdf((4 * k + 1) * abs_max / np.sqrt(n))
        terms_two.append(sst.norm.cdf((4 * k + 3) * abs_max / np.sqrt(n)) - sub)

    p_val = 1.0 - np.sum(np.array(terms_one))
    p_val += np.sum(np.array(terms_two))
    return p_val

#Reference Code: https://gist.github.com/StuartGordonReid/ff86c5a895fa90b0880e
def approximate_entropy_test(block, pattern_length=8):
    n = len(block)

    # Convert binary integers to binary strings
    new_block = ''.join(str(b) for b in block)

    # Add first m+1 bits to the end
    # NOTE: documentation says m-1 bits but that doesn't make sense, or work.
    new_block += new_block[:pattern_length + 1]

    # Get max length one patterns for m, m-1, m-2
    max_pattern = ''
    for i in range(pattern_length + 2):
        max_pattern += '1'

    # Keep track of each pattern's frequency (how often it appears)
    vobs_one = np.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs_two = np.zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

    for i in range(n):
        # Work out what pattern is observed
        vobs_one[int(new_block[i:i + pattern_length], 2)] += 1
        vobs_two[int(new_block[i:i + pattern_length + 1], 2)] += 1

    # Calculate the test statistics and p values
    vobs = [vobs_one, vobs_two]
    sums = np.zeros(2)
    for i in range(2):
        for j in range(len(vobs[i])):
            if vobs[i][j] > 0:
                sums[i] += vobs[i][j] * math.log(vobs[i][j] / n)
        sums[i] /= n
    ape = sums[0] - sums[1]
    chi_squared = 2.0 * n * (math.log(2) - ape)
    p_val = spc.gammaincc(pow(2, pattern_length-1), chi_squared/2.0)
    return p_val

# Reference Code: https://gist.github.com/StuartGordonReid/a514ed478d42eca49568
def linear_complexity_test(block, block_size = 4):
    dof = 6
    piks = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

    t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
    mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2

    num_blocks = int(len(block) / block_size)
    if num_blocks > 1:
        block_end = block_size
        block_start = 0
        blocks = []
        for i in range(num_blocks):
            blocks.append(block[block_start:block_end])
            block_start += block_size
            block_end += block_size

        complexities = []
        for block in blocks:
            complexities.append(berlekamp_massey_algorithm(block))

        t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])
        vg = np.histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
        im = ([((vg[ii] - num_blocks * piks[ii]) ** 2) / (num_blocks * piks[ii]) for ii in range(7)])

        chi_squared = 0.0
        for i in range(len(piks)):
            chi_squared += im[i]
        p_val = spc.gammaincc(dof / 2.0, chi_squared / 2.0)
        return p_val
    else:
        return -1.0


def berlekamp_massey_algorithm(block_data):

    n = len(block_data)
    c = np.zeros(n)
    b = np.zeros(n)
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    int_data = [int(el) for el in block_data]
    while i < n:
        v = int_data[(i - l):i]
        v = v[::-1]
        cc = c[1:l + 1]
        d = (int_data[i] + np.dot(v, cc)) % 2
        if d == 1:
            temp = copy.copy(c)
            p = np.zeros(n)
            for j in range(0, l):
                if b[j] == 1:
                    p[j + i - m] = 1
            c = (c + p) % 2
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b = temp
        i += 1
    return l

def universal(block: list[int]):

    bin_str = "".join(str(bit) for bit in block)

    # The below table is less relevant for us traders and markets than it is for security people
    n = len(block)

    pattern_size = 5
    if n >= 387840:
        pattern_size = 6
    if n >= 904960:
        pattern_size = 7
    if n >= 2068480:
        pattern_size = 8
    if n >= 4654080:
        pattern_size = 9
    if n >= 10342400:
        pattern_size = 10
    if n >= 22753280:
        pattern_size = 11
    if n >= 49643520:
        pattern_size = 12
    if n >= 107560960:
        pattern_size = 13
    if n >= 231669760:
        pattern_size = 14
    if n >= 496435200:
        pattern_size = 15
    if n >= 1059061760:
        pattern_size = 16

    if 5 < pattern_size < 16:
        # Create the biggest binary string of length pattern_size
        ones = ""
        for i in range(pattern_size):
            ones += "1"

        # How long the state list should be
        num_ints = int(ones, 2)
        vobs = np.zeros(num_ints + 1)

        # Keeps track of the blocks, and whether were are initializing or summing
        num_blocks = math.floor(n / pattern_size)
        init_bits = 10 * pow(2, pattern_size)
        test_bits = num_blocks - init_bits

        # These are the expected values assuming randomness (uniform)
        c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
        variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
        expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                    10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
        sigma = c * math.sqrt(variance[pattern_size] / test_bits)

        cumsum = 0.0
        for i in range(num_blocks):
            block_start = i * pattern_size
            block_end = block_start + pattern_size
            block_data = bin_str[block_start: block_end]
            # Work out what state we are in
            int_rep = int(block_data, 2)

            # Initialize the state list
            if i < init_bits:
                vobs[int_rep] = i + 1
            else:
                initial = vobs[int_rep]
                vobs[int_rep] = i + 1
                cumsum += math.log(i - initial + 1, 2)

        # Calculate the statistic
        phi = float(cumsum / test_bits)
        stat = abs(phi - expected[pattern_size]) / (float(math.sqrt(2)) * sigma)
        p_val = spc.erfc(stat)
        return p_val
    else:
        return -1.0


def main():
    file_path = 'moves (6).txt'
    moves = read_moves_from_file(file_path)
    print(moves)

    block_length = 8

    non_hashed_seq = convert_moves_to_binary(moves)
    print(non_hashed_seq)
    non_hashed_block = divide_binary_sequence(non_hashed_seq, block_length)
    print(non_hashed_block)

    binary_sequence = extractor(moves)
    print(binary_sequence)
    blocks = divide_binary_sequence(binary_sequence, block_length)
    print(blocks)

    nist_tests = [frequency_test, runs_test, serial_test, random_excursion_test, random_excursion_variant_test, cumulative_sums_test, approximate_entropy_test, linear_complexity_test]
    test_names = ["Frequency Test", "Runs Test", "Serial Test", "Random Excursion Test", "Random Excursion Variant Test", "Cumulative Sums Test", "Approximate Entropy Test", "Linear Complexity Test"]

    total_tests = 0
    passed_tests = 0

    un_hashed_total_tests = 0
    un_hashed_passed_tests = 0

    test_counters = [0] * len(nist_tests)
    un_hashed_test_counters = [0] * len(nist_tests)

    for i, block in enumerate(blocks):
        print(f"Block {i + 1}:")
        for j, test in enumerate(nist_tests):
            total_tests += 1
            p_value = test(block)
            if p_value > 0.05:
                print(f"  {test_names[j]}: passed")
                passed_tests += 1
                test_counters[j] += 1
            else:
                print(f"  {test_names[j]}: failed")

    for i, block in enumerate(non_hashed_block):
        print(f"Un-Hashed Block {i + 1}:")
        for j, test in enumerate(nist_tests):
            un_hashed_total_tests += 1
            p_value = test(block)
            if p_value > 0.05:
                print(f"  {test_names[j]}: passed")
                un_hashed_passed_tests += 1
                un_hashed_test_counters[j] += 1
            else:
                print(f"  {test_names[j]}: failed")

    print('--------------------------------------------------------------------------')
    percentage_passed = (passed_tests / total_tests) * 100
    print(f'Percentage of NIST tests passed: {percentage_passed}%')

    print('--------------------------------------------------------------------------')
    non_hashed_percentage_passed = (un_hashed_passed_tests / un_hashed_total_tests) * 100
    print(f'Percentage of NIST tests passed (Un-Hashed): {non_hashed_percentage_passed}%')

    print('--------------------------------------------------------------------------')
    print("Individual test pass counts: ")
    for i, test_name in enumerate(test_names):
        print(f"{test_name}: {test_counters[i]}")

    print('--------------------------------------------------------------------------')
    print("Individual un-hashed test pass counts: ")
    for i, test_name in enumerate(test_names):
        print(f"{test_name}: {un_hashed_test_counters[i]}")

if __name__ == '__main__':
    main()






