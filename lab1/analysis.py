import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def Getr2(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in X.index:
        diffX = X[i] - xBar
        diffY = Y[i] - yBar
        SSR += (diffX * diffY)
        varX += diffX**2
        varY += diffY**2
    SST = math.sqrt(varX * varY)
    return (SSR / SST)**2


data_fixedE = pd.DataFrame(columns=['Gain', 'Exposure', 'Value'])
data_fixedG = pd.DataFrame(columns=['Gain', 'Exposure', 'Value'])

for line in open('./data/data_fixedE.txt', 'r'):
    if line == '' or line == '\n':
        break
    else:
        a, b, c = map(float, line.split())
    data_fixedE = data_fixedE.append(
        {'Gain': a, 'Exposure': b, 'Value': c}, ignore_index=True)

for line in open('./data/data_fixedG.txt', 'r'):
    if line == '' or line == '\n':
        break
    else:
        a, b, c = map(float, line.split())
    data_fixedG = data_fixedG.append(
        {'Gain': a, 'Exposure': b, 'Value': c}, ignore_index=True)


print(data_fixedE.groupby('Exposure', as_index=False).apply(lambda data: pd.Series(
    {'r^2^': Getr2(data['Gain'], data['Value'])})).to_markdown(index=False))
print()
print(data_fixedG.groupby('Gain', as_index=False).apply(lambda data: pd.Series(
    {'r^2^': Getr2(data['Exposure'], data['Value'])})).to_markdown(index=False))

for G in data_fixedG['Gain'].unique():
    x = np.array(data_fixedG.query('Gain=='+str(G))['Exposure'])
    y = np.array(data_fixedG.query('Gain=='+str(G))['Value'])
    z1 = np.polyfit(x, y, 2)
    p1 = np.poly1d(z1)
    print('Gain = ', G, ':\n', p1, sep='')
    plt.figure(figsize=(6, 6))
    yvals = p1(x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('Exposure')
    plt.ylabel('GrayScale')
    plt.legend(loc=4)
    plt.title('GrayScale-Exposure')
    # plt.show()
    plt.savefig('./assets/Poly_Gain_'+str(int(G))+'.svg')

for G in data_fixedE['Exposure'].unique():
    x = np.array(data_fixedE.query('Exposure=='+str(G))['Gain'])
    y = np.array(data_fixedE.query('Exposure=='+str(G))['Value'])
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    plt.figure(figsize=(6, 6))
    print('Exposure = ', G, ':\n', p1, sep='')
    yvals = p1(x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('Gain')
    plt.ylabel('GrayScale')
    plt.legend(loc=4)
    plt.title('GrayScale-Gain')
    # plt.show()
    plt.savefig('./assets/Poly_Exposure_'+str(int(G))+'.svg')

plt.figure(figsize=(6, 6))
fixed_Exposeure = sns.lmplot(
    data=data_fixedE,
    x='Gain', y='Value', hue='Exposure'
).fig.suptitle('Fixed Exposure', size=20)
plt.savefig('./assets/Linear_Exposure.svg')
# plt.show()

plt.figure(figsize=(6, 6))
fixed_Gain = sns.lmplot(
    data=data_fixedG,
    x='Exposure', y='Value', hue='Gain'
).fig.suptitle('Fixed Gain', size=20)
plt.savefig('./assets/Linear_Gain.svg')
# plt.show()
