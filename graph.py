import matplotlib.pyplot as plt

# 1. Define los datos para el eje X y el eje Y
# Puedes cambiar estos valores por los tuyos propios
eje_x = [1, 2, 3, 4, 5]
eje_y = [
    0.058091286307053944,
    0.15510204081632653,
    0.3511450381679389,
    0.4457831325301205,
    0.5076335877862596,
]

# 2. Crea la gráfica de puntos unidos por líneas
# 'marker="o"' añade los puntos visibles en cada coordenada
# 'linestyle="-"' une los puntos con una línea continua
# 'color="blue"' define el color de la línea
# 'linewidth=2' define el grosor de la línea
plt.plot(eje_x, eje_y, marker="o", linestyle="-", color="blue", linewidth=2)

# 3. Añade los títulos y etiquetas para los ejes
plt.title("QBER vs distance travelled", fontsize=14, fontweight="bold")
plt.xlabel("Distance travelled (km)", fontsize=12)
plt.ylabel("QBER", fontsize=12)

# 4. Añade una cuadrícula opcional para facilitar la lectura
plt.grid(True, linestyle="--", alpha=0.6)

# 5. Muestra la gráfica en pantalla
plt.show()
