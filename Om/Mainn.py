from img_process import extract as extract_img_grid
from cnn import run as create_and_save_Model
from predict import extract_number_image as sudoku_extracted
from solver import  ConstarintBacktracking as solve_sudoku

def display_gameboard(sudoku):
    for i in range(len(sudoku)):
        if i % 3 == 0:
            if i == 0:
                print(" ┎─────────┰─────────┰─────────┒")
            else:
                print(" ┠─────────╂─────────╂─────────┨")

        for j in range(len(sudoku[0])):
            if j % 3 == 0:
                print(" ┃ ", end=" ")

            if j == 8:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", " ┃")
            else:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", end=" ")

    print(" ┖─────────┸─────────┸─────────┚")

def main():
    
    image_grid = extract_img_grid()
    print("Image Grid extracted")

    sudoku = sudoku_extracted(image_grid)
    print("Extracted and predict digits in the Sudoku")

    print("\n\nSudoku:")
    display_gameboard(sudoku)

    print("\nSolving the Sudoku...\n")
    solvable, solved = solve_sudoku(sudoku)

    if(solvable):
        print("\nSolved Sudoku:")
        display_gameboard(solved)

    print("Program End")




if __name__ == '__main__':
    main()