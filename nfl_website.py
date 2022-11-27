#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import base64
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title('NFL Football Stats For My Understanding')

st.markdown("""
This app performs simple webscraping of NFL Football player stats data & Predicted Wins Vs. Actual Wins!!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")


# load dataset

url_csv = "https://github.com/alemachado24/NFL/blob/ad910a8343f728282ba33b6c936c512eae0cac0c/season_2021.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url_csv).content

nfl = pd.read_csv(io.StringIO(download.decode('utf-8')))

# nfl = pd.read_csv('/Users/am/Desktop/AleClasses/NFL/season_2021.csv')
# nfl = pd.read_csv(r'https://github.com/alemachado24/NFL/blob/ad910a8343f728282ba33b6c936c512eae0cac0c/season_2021.csv')
# https://github.com/alemachado24/NFL/blob/ad910a8343f728282ba33b6c936c512eae0cac0c/season_2021.csv

# inspect first few rows
# nfl.head()

st.header('Display Player Stats for 2021 Season')
# st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(nfl)

# nested dictionary to encode alphanumeric values to numeric values
result_encoder = {'result': {'W': 1, 'T': 0, 'L': 0}}

# encode result column using encoder
nfl.replace(result_encoder, inplace=True)

# check result value counts
# nfl.result.value_counts()

graph_elements = ['1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense',
       'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense',
       'RushY_defense', 'TO_defense']

# Sidebar - Graph Elements
# selected_team = st.sidebar.multiselect('Team', symbols, symbols)
selected_stat = st.sidebar.multiselect('Stats', graph_elements,default = graph_elements[4])

st.header('Visualize the stats for 2021 Season')
# change stat to view plot
stat = selected_stat[0] #'1stD_offense'

st.text(selected_stat[0])

# box plot of stat
stat_plot = sns.boxplot(x='result', y=stat, data=nfl)

st.set_option('deprecation.showPyplotGlobalUse', False)
# plot labels
stat_plot.set_xticklabels(['loss/tie','win'])
st.pyplot(x='result', y=stat, data=nfl)
# list feature names
print(nfl.columns[8:])


# select feature variables to be scaled
features = nfl.iloc[:,8:]
scaler = StandardScaler()

# fit the transformer to the features
scaler.fit(features)

# transform and save as X
X = scaler.transform(features)

# save result variable as y
y = nfl['result']

# create train-test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# create the classifier
lrc = LogisticRegression()

# fit classifier to the training data
lrc.fit(X_train, y_train)

# predict with the classifier using the .predict() function
y_pred = lrc.predict(X_test)

# view the model accuracy with the accuracy_score() function
accuracy_score(y_pred,y_test)

# create a list of penalties
penalties = ['l1', 'l2']
# create a list of values for C
C = [0.01, 0.1, 1.0, 10.0, 1000.0]

for penalty in penalties:
    for c in C:

        # instantiate the classifier
        lrc_tuned = LogisticRegression(penalty=penalty, C=c, solver='liblinear')

        # fit the classifier to the training data
        lrc_tuned.fit(X_train, y_train)
        
        # predict with the classifier using the .predict() function
        y_pred = lrc_tuned.predict(X_test)

        # view the model accuracy with the accuracy_score() function
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_rd = round(accuracy*100,1)
        
        # print accuracy for each combination of penalty and C
        
        st.text(f'Accuracy: {accuracy_rd}% | penalty = {penalty}, C = {c}')
        
#Missing to print the max accuracy instead of all of them !!!

#         st.text(f'Accuracy: {max(accuracy_rd)}% | penalty = {penalty}, C = {c}')
# st.text('Optimal Penalty & C Value:')

# max_value = None
# for n in accuracy_score(y_test, y_pred):
#     if max_value is None or n > max_value: max_value = n
# st.text(f'Accuracy: {max_value}%')

#Missing to set this numbers to a variable dependable from above same for test sizes!!!
# optimal penalty and C
penalty = 'l1'
C = 0.1

# create a list of test_sizes
test_sizes = [val/100 for val in range(20,36)]

for test_size in test_sizes:

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # instantiate the classifier
    lrc_tts = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

    # fit the classifier to the training data
    lrc_tts.fit(X_train, y_train)

    # predict with the classifier using the .predict() function
    y_pred = lrc_tts.predict(X_test)

    # view the model accuracy with the accuracy_score() function
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_rd = round(accuracy*100,1)
    
    # print accuracy for each combination of penalty and test size
    st.text(f'Accuracy: {accuracy_rd}% | test size = {test_size}')
    
    
# set the test size and hyperparameters
test_size = 0.25
penalty = 11
C = 0.1

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# instantiate the classifier
optLr = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

# fit the classifier to the training data
lrc.fit(X_train, y_train)


# get importance
importance = abs(lrc.fit(X_train, y_train).coef_[0])

# visualize feature importance
sns.barplot(x=features.columns, y=importance)

# add labels and titles
# plt.suptitle('Feature Importance for Logistic Regression')
# plt.xlabel('Score')
# plt.ylabel('Stat')
# plt.show()

st.header('Feature Importance for Logistic Regression for 2021 Season')
chart_data = pd.DataFrame(
    importance,
    features.columns)

st.bar_chart(chart_data)


# summarize feature importance
for i,v in enumerate(importance.round(2)):
    st.text(f'Feature: {features.columns[i]}, Score: {v}')

#side bars
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))


# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2019/rushing.htm
symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']

# st.text([x.lower() for x in [symbols]])

# Sidebar - Team selection
all_lower_sym = [x.lower() for x in symbols]
# st.text(all_lower_sym)
sorted_unique_team = sorted(all_lower_sym)
# selected_team = st.sidebar.multiselect('Team', symbols, symbols)
selected_team = st.sidebar.multiselect('Team', sorted_unique_team,default = all_lower_sym[11])




# st.text(symbols[team_names.index(selected_team)].lower())

@st.cache
def get_new_data(team, year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    
    - team : team name (str)
    - year : year (int)
    '''
    # pull data
    url = f'https://www.pro-football-reference.com/teams/{selected_team[0]}/{selected_year}.htm'
    html = requests.get(url).text
#     st.text(url)

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    table = soup.find('table', id='games')
    tablerows = table.find_all('tr')[2:]
    data = []

    for tablerow in tablerows:
        data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

    df = pd.DataFrame(data)


    # subset
    index = [0,1,4,8,9,10] + list(range(11,21))
    new_data = df.iloc[:,index].copy()

    # rename columns
    col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score', '1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense', 'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense', 'RushY_defense', 'TO_defense']
    new_data.columns = col_names

    # encode results
    result_encoder = {'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA},
                     'TO_offense' : {'' : 0},
                     'TO_defense' : {'' : 0}}
    new_data.replace(result_encoder, inplace=True)

    # remove future dates
    new_data = new_data[new_data.result.notnull()]

    # add week variable back
    week = list(range(1,len(new_data)+1))
    new_data.insert(0, 'week', week)
    
    # add team name
    tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
    new_data.insert(0, 'team_name', tn_col)

    # return a dataframe object
    if type(new_data) == pd.Series:
        new_data = new_data.to_frame().T
        return new_data.reset_index(drop=True)
    else:
        return new_data.reset_index(drop=True)



st.text(selected_team[0])
st.text(selected_year)

# st.text(list(new_data.team_name.unique()))

# # Sidebar - Position selection
# # unique_pos = ['RB','QB','WR','FB','TE']
# unique_pos = symbols
# selected_pos = st.sidebar.multiselect('Teams', symbols, symbols)

# team = 'Minnesota Vikings'
# year = 2022

st.header(f'Display Player Stats of {selected_team[0].upper()} in {selected_year}')
# st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
# st.dataframe(new_data)
new_data = get_new_data(team=selected_team[0].upper(), year=selected_year)

# Filtering data
# df_selected_team = new_data[(new_data.team_name.isin(selected_team)) & (new_data.year.isin(selected_year))]
# df_selected_team = new_data[(selected_team) & (selected_year)]
# new_data.head()
st.dataframe(new_data)
# st.dataframe(df_selected_team)

# st.text('test')


# select just the game stats
new_X = new_data.loc[:,features.columns]

# standardize using original data's scaling
new_X_sc = scaler.transform(new_X)

# get new predictions
new_preds = lrc.fit(X_train, y_train).predict(new_X_sc)

# get actual results and set type to float
new_results = new_data['result'].astype(float)

# get accuracy score for new data
acc_score = accuracy_score(new_results, new_preds)

# select only game data
col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
game_data = new_data.loc[:,col_names]
# create comparison table
comp_table = game_data.assign(predicted = new_preds,
                              actual = new_results.astype(int))


# st.header('Predicted Wins vs Actual Wins')
# print title and table
st.header(f'Predicted Wins vs Actual Wins for {selected_team[0].upper()} in {selected_year}')
comp_table

# print accuracy
st.text(f'\nCurrent Accuracy Score: ' + str(round(acc_score*100,1)) + '%')



# # Download NBA player stats data
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# # Heatmap
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_selected_team.to_csv('output.csv',index=False)
#     df = pd.read_csv('output.csv')

#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot()




