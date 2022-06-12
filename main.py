# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
# %%
# %%
plt.style.use("seaborn-darkgrid")
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["text.color"] = "k"
mpl.rcParams["figure.dpi"] = 200

directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)
# Exercise: exploratory analysis
# %%
Path("plots").mkdir(parents=True, exist_ok=True)
mpl.rcParams["figure.dpi"] = 150
sns.set_theme()
# %%
titanic = sns.load_dataset("titanic")
titanic.columns
titanic.describe()
titanic.info()
# %%
corrmat = titanic.corr()
# %%
colormap = plt.cm.Blues
f, ax = plt.subplots(figsize=(11, 9))
heatmap = sns.heatmap(titanic.corr(), cmap=colormap, annot=True, linewidths=0.2)
f.savefig("plots/heatmap.png")
plt.clf()
# %%
titanic.survived.value_counts()
# %%
titanic.pclass.value_counts()

# %%
catplot = sns.catplot(x="sex", y="survived", hue="pclass", kind="point", data=titanic)
catplot.savefig("plots/catplot1.png")
# %%
catplot2 = sns.catplot(x="pclass", y="survived", hue="sex", kind="point", data=titanic)
catplot2.savefig("plots/catplot2.png")
# %%
f, ax = plt.subplots(figsize=(11, 9))
violin = sns.violinplot(x="pclass", y="fare", hue="survived", data=titanic)
f.savefig("plots/violin.png")

# %% Exercise: integral
import numpy as np

dx = 0.01
x_1 = 1.0
x_0 = 0.0


# %%
def f(x):
    return x * np.cos(x**2)


# %%
x = x_0
integral = 0
while x < x_1:
    integral += dx * f(x)
    x += dx
print(f"{integral=}")

# Exercise: regressions 
cars = pd.read_csv('assets/imports-85.txt')
cars['price'] = pd.to_numeric(cars.price)
cars['horsepower'] = pd.to_numeric(cars.horsepower)

cars['make'].value_counts()
# %%
plt.hist(cars[['horsepower']], rwidth=1, bins=30, label='length')
plt.legend()
plt.savefig(directory + '/horsepower_hist')
plt.clf()
# %%
formula = 'price ~ horsepower'
model = smf.ols(formula, data = cars)
results = model.fit()
results.summary()

# %%
inter = results.params['Intercept']
slope = results.params['horsepower']
print(f'{inter=}')
print(f'{slope=}')

# %%
plt.scatter(x='horsepower',y='price', data=cars)
plt.xlabel('horsepower')
plt.ylabel('price')
plt.legend()
plt.savefig(directory + '/price_horsepower_scatter.png')
plt.clf()

# %%
formula = 'price ~ horsepower + length + width'
model = smf.ols(formula, data = cars)
results = model.fit()
results.summary()
inter = results.params['Intercept']
slope = results.params['horsepower']
print(f'{inter=}')
print(f'{slope=}')

# %%
gas = cars[cars['fuel-type']=="gas"]
diesel = cars[cars['fuel-type']=="diesel"]
# %%
plt.boxplot(x=[gas.price, diesel.price])
plt.xlabel('fuel-type')
plt.ylabel('price')
plt.legend()
plt.savefig(directory + '/fuel_boxplot.png')
plt.clf()
# %%
# %%
cars['gas'] = (cars['fuel-type'] == 'gas') * 1
formula = 'gas ~ price + length + horsepower'
model = smf.logit(formula, data=cars)
results = model.fit()
results.summary()
# %%
new = pd.DataFrame([[24000, 190, 120]], columns=['price', 'length', 'horsepower'])
y = results.predict(new)
print(f'The chances of this car being gas are {y[0]}')

# Exercise: bayesian statistics 
# %%
import pandas as pd
# %%
table = pd.DataFrame(index=['normal', 'trick'])
table['prior'] = 1/2, 1/2
table['likelihood'] = 1/2, 1
table['unnorm'] = table.prior * table.likelihood
prob_data = table.unnorm.sum()
table['posterior'] = table['unnorm'] / prob_data
table.head()

# %%
elvis = pd.DataFrame(index=['identical', 'fraternal'])
elvis['prior'] = 1/3, 2/3
elvis['likelihood'] = 1, 1/2
elvis['unnorm'] = elvis.prior * elvis. likelihood
prob_data = elvis.unnorm.sum()
elvis['posterior'] = elvis['unnorm'] / prob_data
elvis.head()
# %%
# %%
def prob(proposition):
    return proposition.mean()

# %%
def conditional(proposition, given):
    return prob(proposition[given])

# %%
titanic = sns.load_dataset('titanic')
titanic.head()
# %%
titanic['first'] = titanic['class'] == "First"
conditional(titanic.survived, given = (titanic['first'] & ~titanic['alone']))
conditional(titanic.survived, given = (titanic['first'] & titanic['alone']))
# %%