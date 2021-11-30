def ConstarintBacktracking(sudoku):
    import constraint
    problem = constraint.Problem()

    for i in range(1, 10):
        problem.addVariables(range(i * 10 + 1, i * 10 + 10), range(1, 10))
    
    for i in range(1, 10):
        problem.addConstraint(constraint.AllDifferentConstraint(), range(i * 10 + 1, i * 10 + 10))
    
    for i in range(1, 10):
        problem.addConstraint(constraint.AllDifferentConstraint(), range(10 + i, 100 + i, 10))
    
    for i in [1, 4, 7]:
        for j in [1, 4, 7]:
            square = [10 * i +j, 10 * i + j +1, 10 * i + j +2,
                      10 *( i + 1 ) +j, 10 *( i + 1 ) + j +1, 10 *( i + 1 ) + j +2,
                      10 *( i + 2 ) +j, 10 *( i + 2 ) + j +1, 10 *( i + 2 ) + j +2]
            problem.addConstraint(constraint.AllDifferentConstraint(), square)
    
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                def constraint(variable_value, value_in_table = sudoku[i][j]):
                    if variable_value == value_in_table:
                        return True
    
                problem.addConstraint(constraint, [(( i +1 ) *10 + ( j +1))])
    
    solutions = problem.getSolutions()
    sudoku_solv =[[0 for x in range(9)] for y in range(9)]
    solavble = False

    if len(solutions) == 0:
        print("No solutions found.")
    else:
        solution = solutions[0]
        solavble = True

        for i in range(1, 10):
            for j in range(1, 10):
                sudoku_solv[i - 1][j - 1] = (solution[i * 10 + j])

        # print(sudoku_solv)

    return solavble, sudoku_solv