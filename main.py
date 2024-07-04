import openpyxl as openxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readXl (xl_file):  # funcao para ler o arquivo .xl (apenas a primeira coluna da linha 2 ate 501 é lida)
    wb = openxl.load_workbook(xl_file)
    ws = wb.active
    data = []
    for row in ws.iter_rows(min_row=2, max_row=501, min_col=1, max_col=1, values_only=True):    # parametros para ler-se somente a primeira coluna com dados brutos em rol
        data.append(row[0])

    return data

xl_file = "C:\\Users" # FIXME

def dataPlot (data_set):
    plt.plot(data_set)
    plt.title('Data from Excel Column A2:A500')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def frequencyTable (data):
    n = len(data)
    classes = int(np.round(np.sqrt(n)))
    max_amplitude = data[n - 1] - data[0]
    classes_amp = int(np.ceil((max_amplitude / classes)))
    bins = np.linspace(data[0], data[-1] + classes_amp, classes + 1)
    histogram = np.histogram(data, bins=bins)
    return print(histogram)

def main ():
    data = readXl(xl_file)
    print(data)
    frequencyTable(data)
    dataPlot(data)

main()
