import pandas as pd
import numpy as np

data = pd.read_csv('../data/mlbootcamp5_train.csv', sep=';', index_col='id')

def identify_gender():
    if data[data['gender'] == 1]['height'].mean() > \
       data[data['gender'] == 2]['height'].mean():
        print(len(data[data['gender'] == 1]), 'мужчин ',
              len(data[data['gender'] == 2]), 'женщин')
    else:
        print(len(data[data['gender'] == 1]), 'женщин',
              len(data[data['gender'] == 2]), 'мужчин')

def identify_alco():
    if len(data[(data['gender'] == 1) & (data['alco'] == 0)]) / len(data[data['gender'] == 1]) > \
       len(data[(data['gender'] == 2) & (data['alco'] == 0)]) / len(data[data['gender'] == 2]):
        print('Мужчины')
    else:
        print('Женщины')
    print(pd.crosstab(data['gender'], data['alco']))

def identify_proc_smoke():
    proc_smoke_famale = data[data['gender'] == 1]['smoke'].mean()
    proc_smoke_male = data[data['gender'] == 2]['smoke'].mean()
    print(round(proc_smoke_male / proc_smoke_famale))

def indetify_median_smoke():
    b = data[data['smoke'] == 1]['age']/365 * 12
    v = data[data['smoke'] == 0]['age']/365 * 12
    print(round(b.median() - v.median()))

def calculation_risk_death():
    data['age_years'] = round(data['age'] / 365)
    smoke_male = data[(data['smoke'] == 1) & (data['age_years'] >= 60) & (data['age_years'] <= 64) & (data['gender'] == 2)]
    first_sample = smoke_male[(data['cholesterol'] == 1) & (data['ap_hi'] < 120)]['cardio'].mean()
    second_sample = smoke_male[(data['cholesterol'] == 3) & (data['ap_hi'] >= 160) & (data['ap_hi'] < 180)]['cardio'].mean()
    #print(first_sample)
    print(second_sample / first_sample)

def bmi():
    data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
    print(data['bmi'].median(), 'медианный')
    print(data[data['gender'] == 1]['bmi'].mean(),
          data[data['gender'] == 2]['bmi'].mean(), 'женщина, мужик')
    print(data[(data['cardio'] == 1)]['bmi'].mean(),
          data[(data['cardio'] == 0)]['bmi'].mean(), 'больной, здоровый')
    print(data[(data['cardio'] == 0) & (data['alco'] == 0) & (data['gender'] == 2)]['bmi'].mean(),
          data[(data['cardio'] == 0) & (data['alco'] == 0) & (data['gender'] == 1)]['bmi'].mean(), 'мужик, баба')
    #print(data)

def cleaning():
    clean_data = data[data['ap_hi'] >= data['ap_lo']]
    clean_data = clean_data[(clean_data['height'] >= clean_data['height'].quantile(.025)) & (
            clean_data['height'] <= clean_data['height'].quantile(.975))]
    clean_data = clean_data[(clean_data['weight'] >= clean_data['weight'].quantile(.025)) & (
            clean_data['weight'] <= clean_data['weight'].quantile(.975))]
    print(((len(data) - len(clean_data)) / len(data)) * 100)

#identify_gender()
#identify_alco()
#identify_proc_smoke()
#indetify_median_smoke()
#calculation_risk_death()
#bmi()
#cleaning()
#print(data)


