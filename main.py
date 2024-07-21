import openpyxl as openxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import scipy.stats as stats
import statsmodels.api as sm 
import pylab as py 

def readXl(xl_file):  
    # Function to read the Excel file (only the first column from row 2 to 501)
    wb = openxl.load_workbook(xl_file)
    ws = wb.active
    data = []
    for row in ws.iter_rows(min_row=2, max_row=501, min_col=1, max_col=1, values_only=True):
        data.append(row[0])
    return sorted(data)

def IsNormal (data):
    # Function to check if the data is normally distributed
    stat, p_value = stats.shapiro(data)
    alpha = 0.05
    print(f'Estatística do teste: {stat}')
    print(f'Valor p: {p_value}')
    if p_value > alpha:
        return print('Não há evidência de que os dados não sejam normalmente distribuídos')
    else:
        return print('Há evidência de que os dados não sejam normalmente distribuídos')
    
    
def dataPlotHist(data, frequency):
    n, bins, patches = plt.hist(data, bins=25, alpha=0.6, color='b', edgecolor='black')

    plt.title('Médias de VMs ociosas X frequência')
    plt.xlabel('Média de VMs ociosas')
    plt.ylabel('Frequência')

    max_freq = max(n)
    y_ticks = np.arange(0, max_freq + 1, step=int(max_freq / 10))
    plt.yticks(y_ticks)

    plt.show()

def dataplotNormal(data):    
    # Q-Q plot
    standardized_data = (data - np.mean(data)) / np.std(data)
    sm.qqplot(standardized_data, line='45')

    plt.xlabel('Quantis teóricos')
    plt.ylabel('Média de VMs ociosas')
    plt.title("Q-Q plot")

    plt.show()

def frequencyTable(data):
    data = np.sort(data)
    n = len(data)
    classes = int(np.round(np.sqrt(n)))
    if classes > 20:
        classes = 7
    max_amplitude = data[-1] - data[0]
    classes_amp = int(np.ceil(max_amplitude / classes))
    bins = [data[0]]
    for i in range(1, classes + 1):
        bins.append(bins[-1] + classes_amp)
    bins = np.array(bins)
    
    freq, edges = np.histogram(data, bins=bins)
    print(freq, edges)
    return freq, edges

def modeGrouped (freq, classes):
    list_freq = freq.tolist()
    list_classes = classes.tolist()
    max_freq = list_freq.index(max(list_freq))
    max_classes = list_classes[max_freq]
    delta_1 = list_freq[max_freq] - list_freq[max_freq - 1]
    delta_2 = list_freq[max_freq] - list_freq[max_freq + 1]
    mode = (delta_1 / (delta_1 + delta_2)) * (list_classes[list_classes.index(max_classes) + 1] - max_classes)
    mode += max_classes
    return mode
    
def temporalPlot(n_days, data_size, data):
    days = np.linspace(1, n_days, data_size)
    data_f = pd.DataFrame({
        'Days': days,
        'VMs': data
    })
    data_grouped = data_f.groupby('Days')
    plt.plot(data_grouped['Days'], data_grouped['VMs'], marker='o', linestyle='-')
    plt.grid(True)
    plt.show()

def BoxPlot(data):
    plt.boxplot(data)
    plt.show()

def calculateQuartile(freq, classes):
    freq = freq.tolist()
    classes = classes.tolist()
    n = sum(freq)

    Q1_pos = 0.25 * n
    Q2_pos = 0.50 * n
    Q3_pos = 0.75 * n

    quartiles = []

    def findQuartile(position):
        cumulative_freq = 0
        for i, f in enumerate(freq):
            cumulative_freq += f
            if cumulative_freq >= position:
                lower_class_limit = classes[i]
                upper_class_limit = classes[i + 1]
                cumulative_freq -= f
                quartile_value = lower_class_limit + ((position - cumulative_freq) / f) * (upper_class_limit - lower_class_limit)
                return quartile_value

    Q1 = findQuartile(Q1_pos)
    Q2 = findQuartile(Q2_pos)
    Q3 = findQuartile(Q3_pos)

    quartiles.extend([Q1, Q2, Q3])
    return quartiles

def calculateCV(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = (std_dev / mean) * 100
    return cv

def calculateCurtose(data):
    curtose = stats.kurtosis(data)
    return curtose

def main():
    xl_file = "seminario_estatistica_(VM).xlsx"
    data = readXl(xl_file)
    freq, edges = frequencyTable(data)
    dataPlotHist(data, freq)
    dataplotNormal(data)
    BoxPlot(data)
    IsNormal(data)

    quartiles = calculateQuartile(freq, edges)
    cv = calculateCV(data)
    curtose = calculateCurtose(data)

    print(f"média {np.round(np.average(data), 4)}")
    print(f"mediana {np.round(np.median(data), 4)}")
    print(f"variancia {np.round(statistics.pvariance(data), 4)}")
    print(f"desvio padrao {np.round(statistics.pstdev(data), 4)}")
    print(f"moda {np.round(modeGrouped(freq, edges), 4)}")
    print(f'quartis: Q1 = {np.round(quartiles[0], 4)}, Q2 = {np.round(quartiles[1], 4)}, Q3 = {np.round(quartiles[2], 4)}')
    print(f'coeficiente de variação: {np.round(cv, 4)}%')
    print(f'curtose: {np.round(curtose, 4)}')

main()
