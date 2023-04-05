
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

def main():
    file_path = 'games/allgames.txt'
    #read_all_files()
    moves = read_moves_from_file(file_path)
    grid = visualize(moves)
    for row in grid:
        print(row)
    
if __name__ == '__main__':
    main()