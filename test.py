matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

linhas = len(matriz)
colunas = len(matriz[0])
x = linhas - 1
y = 0


for linha in range(linhas):
    for coluna in range(colunas):
        print(matriz[x][y])
        x-=1
    x = linhas - 1
    y+=1
