import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import code
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 1.4
mpl.rcParams['xtick.major.width'] = 1.4
mpl.rcParams['ytick.major.width'] = 1.4

def roads5ClassLabel(iriValue: np.float64, allowNeg=False):
    if 0 <= int(iriValue) <= 7:
        labelName = 0  # 'great'
    elif 7 < int(iriValue) <= 12:
        labelName = 1  # 'good'
    elif 12 < int(iriValue) <= 15:
        labelName = 2  # 'fair'
    elif 15 < int(iriValue) <= 20:
        labelName = 3  # 'poor'
    elif int(iriValue) > 20:
        labelName = 4  # 'bad'
    else:
        if allowNeg:
            labelName = 0
        else:
            labelName = 'invalid'
    return labelName


if __name__=="__main__":
    data = pd.read_csv('images_labels_256_df.csv')
    data = data[data['IRI'] >= 0.000]
    data['classLabel'] = data.apply(lambda row: roads5ClassLabel(row['IRI']), axis=1)
    label_map = {0: 'Great', 1:'Good', 2:'Fair', 3:'Poor', 4: 'Bad'}
    data['classLabel'] = data['classLabel'].map(label_map)
    final_data = data.groupby('classLabel').count().reset_index().rename(columns={'IRI':'Frequency'})[['classLabel', 'Frequency']]
    code.interact(local=locals())
    plt.figure(figsize=(6, 4))
    clrs=['#5e3c99', '#e41a1c', '#0571b0']
    sns.histplot(data=data, x="classLabel", shrink=.9, color=clrs[2])
    plt.yscale("log")
    plt.xlabel('Class Labels')
    plt.ylabel('Count of Class Labels')
    sns.despine()
    plt.minorticks_off()
    plt.savefig('graphs/class_distributions.jpg', dpi=600)
    plt.show()
    #code.interact(local=locals())
