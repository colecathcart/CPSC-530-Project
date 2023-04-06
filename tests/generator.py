import hashlib
import math

def gethash(moves):
    decimal_string = ""
    for move in moves:
        decimal_string += str(move[0])
        decimal_string += str(move[1])
    binary_string = bin(int(decimal_string))[2:]
    sha1string = hashlib.md5(binary_string.encode())
    binary_md5 = bin((int(sha1string.hexdigest(), 16)))[2:130]

    print("Hash: " + binary_md5 + "\n")

def generator(moves):
    count = 0
    entropy = 0
    moveset_128 = []
    for move in moves:
        moveset_128.append(move)
        entropy += 5.77
        #entropy += 6.6
        if entropy > 128:
            gethash(moveset_128)
            count += 1
            entropy = 0
            moveset_128 = []
    print("Total numbers created: " + str(count))
        

def read_all_files():
    gameno = 1
    movesmaster = ""
    for i in range(50):
        with open(("games/game"+str(gameno)+".txt"), 'r') as file:
            movesmaster += file.read()
        gameno += 1
    with open("games/allgames.txt", 'w') as file:
        file.write(movesmaster)

def read_moves_from_file(file_path):
    with open(file_path, 'r') as file:
        #moves_string = file.read().strip()
        moves_string = "".join(file.read().split())

    moves = [(int(moves_string[i]), int(moves_string[i + 1])) for i in range(0, len(moves_string), 2)]
    print(moves)
    print("Total moves: " + str(len(moves)))
    return moves

def visualize(moves):
    grid = [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]
    for move in moves:
        grid[move[0]][move[1]] += 1
    return grid

def shannon_entropy(grid):
    shannon_entropy = 0
    for i in grid:
        for j in i:
            shannon_entropy += (j/2025)*(math.log2((1/2025)))
    return -shannon_entropy

def main():
    file_path = 'games/allgames.txt'
    #read_all_files()
    moves = read_moves_from_file(file_path)
    grid = visualize(moves)
    for row in grid:
        print(row)
    generator(moves)
    print("Shannon entropy: " + str(shannon_entropy(grid)))
    
if __name__ == '__main__':
    main()