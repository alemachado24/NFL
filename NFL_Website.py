#!/usr/bin/env python
# coding: utf-8

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
import io


st.set_page_config(page_title="NFL Stats",page_icon="🏈",)

#This highlight the text in light green
# st.sidebar.success("Select a demo above.")

st.title('NFL Football Stats For My Understanding TESTS')
st.sidebar.markdown("NFL Football Stats")


st.markdown("""
This app performs simple webscraping of NFL Football player stats data & Predicted Wins Vs. Actual Wins
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn,io
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

# load dataset
# Make sure the url is the raw version of the file on GitHub
url_csv = "https://raw.githubusercontent.com/alemachado24/NFL/main/season_2021.csv" 
download = requests.get(url_csv).content
nfl = pd.read_csv(io.StringIO(download.decode('utf-8')),sep=',')

#If you have the file saved in your computer
# nfl = pd.read_csv('/Users/am/Desktop/AleClasses/NFL/season_2021.csv')

# inspect first few rows
# nfl.head()

#Title + Table in Display
st.header('Display Player Stats for 2021 Season')

my_expander = st.expander(label='Click Here to access Stats Dataset for 2021 Season')
with my_expander:
    st.dataframe(nfl)
#     clicked = st.button('Click me!')

#side bars
st.sidebar.header('User Input Features')

graph_elements = ['1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense',
       'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense',
       'RushY_defense', 'TO_defense']

# Sidebar - Graph Elements
selected_stat = st.sidebar.multiselect('Stats', graph_elements,default = graph_elements[4])

st.header('Visualize the stats for 2021 Season by Feautures & Loss, Tie & Win')
# change stat to view plot

try:
    stat = selected_stat[0] #'1stD_offense'
    st.text(f'Feature selected for Bar Plots: {stat}')
#     st.text(selected_stat[0])

    # To Transform Wins Ties And L to numeric numbers
    # nested dictionary to encode alphanumeric values to numeric values
    result_encoder = {'result': {'W': 1, 'T': 0, 'L': 0}}
    # encode result column using encoder
    nfl2 = nfl#.replace(result_encoder, inplace=True)


    row1_1, row1_2 = st.columns(2)

    with row1_1:
#         st.header("1")
        st.write("Win, Loss & Tie")
        stat_plot = sns.boxplot(x='result', y=stat, data=nfl)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(x='result', y=stat, data=nfl)
    #     map(filterdata(data, hour_selected), midpoint[0], midpoint[1], 11)

    with row1_2:
#         st.header("2")
        st.write("Win & Loss/Tie")
        nfl2.replace(result_encoder, inplace=True)
        stat_plot = sns.boxplot(x='result', y=stat, data=nfl2)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        stat_plot.set_xticklabels(['Loss/Tie','Win'])
        st.pyplot(x='result', y=stat, data=nfl2)
    #     map(filterdata(data, hour_selected), la_guardia[0], la_guardia[1], zoom_level)
    
except:
    st.warning('Please pick a stat to visualize the plot')
    

#--------- Train de model with original data:
# select feature variables to be scaled 
features = nfl.iloc[:,8:]
scaler = StandardScaler()

# fit the transformer to the features
scaler.fit(features)

# transform and save as X
X = scaler.transform(features)
# st.table(X)

# save result variable as y
y = nfl['result']
# st.table(y)

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

#Header to check Model Accuracy
st.header('Model Optimal Variables:')


best_penalty=0
best_c=0
best_acc=0

#Check best values for Penalty and C values in the regression
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

        if accuracy >= best_acc:
            best_acc=accuracy
            best_penalty=penalty
            best_c=c
        else:
            best_acc=best_acc

st.text(f'Accuracy improves to: {round(best_acc*100,1)}% by optimizing penalty to {best_penalty} and C to {best_c}')
        

# optimal penalty and C
penalty = best_penalty#'l1'
C = best_c#0.1
# st.text(penalty)
# st.text(C)

# create a list of test_sizes with optimal variables
test_sizes = [val/100 for val in range(20,36)]
best_acc_new=0

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
    
    if accuracy >= best_acc_new:
        best_acc_new=accuracy
        best_size=test_size
    else:
        best_acc_new=best_acc_new
    
# print accuracy for each combination of penalty and test size
st.text(f'Accuracy: {round(best_acc_new*100,1)}% | test size = {best_size}')
    

# set the test size and hyperparameters
test_size = best_size#0.25
penalty = best_penalty#11
C = best_c#0.1

#Train the new modelo with the optimal variables:
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# instantiate the classifier
optLr = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

# fit the classifier to the training data
lrc.fit(X_train, y_train)


# get importance per each feature
importance = abs(lrc.fit(X_train, y_train).coef_[0])

# visualize feature importance
sns.barplot(x=features.columns, y=importance)

#sidebar
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

#Header + Table for Importance
st.header(f'Feature Importance for Logistic Regression for 2021 Season')
chart_data = pd.DataFrame(importance,features.columns)
st.bar_chart(chart_data)


# summarize feature importance
# for i,v in enumerate(importance.round(2)):
#     st.text(f'Feature: {features.columns[i]}, Score: {v}')



# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2019/rushing.htm
symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']

# Sidebar - Team selection
all_lower_sym = [x.lower() for x in symbols]
# st.text(all_lower_sym)
sorted_unique_team = sorted(all_lower_sym)
selected_team = st.sidebar.multiselect('Team', sorted_unique_team,default = sorted_unique_team[11])

# try:
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
    #st.text(url)

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    table = soup.find('table', id='games')
    tablerows = table.find_all('tr')[2:]
    data = []

    for tablerow in tablerows:
        data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

    df = pd.DataFrame(data)


    # subset (columns that I want)
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
#     new_data = new_data[new_data.result.notnull()]
#     new_data = new_data[new_data.result.fillna(0, inplace=True)]
#     result.fillna(0, inplace=True)

    # add week variable back
    week = list(range(1,len(new_data)+1))
    new_data.insert(0, 'week', week)

    # add team name
    tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
    new_data.insert(0, 'team', tn_col)

    # return a dataframe object
    if type(new_data) == pd.Series:
        new_data = new_data.to_frame().T
        return new_data.reset_index(drop=True)
    else:
        return new_data.reset_index(drop=True)


# st.header(f'Display Player Stats of {selected_team[0].upper()} in {selected_year}')
new_data = get_new_data(team=selected_team[0].upper(), year=selected_year)
# st.dataframe(new_data)

#to get new X with the new data I need to exclude games that has not happened
new_data2=new_data[new_data.result.notnull()]

# select just the game stats
new_X = new_data2.loc[:,features.columns]

# standardize using original data's scaling
new_X_sc = scaler.transform(new_X)

# get new predictions
new_preds = lrc.fit(X_train, y_train).predict(new_X_sc)

# get actual results and set type to float
new_results = new_data2['result'].astype(float)
# st.dataframe(new_results)
# st.text('predicted')
# st.dataframe(new_preds)

# get accuracy score for new data
acc_score = accuracy_score(new_results, new_preds)

# select only game data
col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
game_data = new_data2.loc[:,col_names]
# create comparison table
comp_table = game_data.assign(predicted = new_preds,
                              actual = new_results.astype(int))

# rename columns
col_names = ['Day', 'Date', 'Result', 'Opponent', 'Team Score', 'Opp Score','Predicted','Actual']
comp_table.columns = col_names

# st.header('Predicted Wins vs Actual Wins')
# print title and table
st.header(f'Predicted Wins vs Actual Wins for {selected_team[0].upper()} in {selected_year}')

my_expander2 = st.expander(label=(f'Click Here to display Stats of {selected_team[0].upper()} in {selected_year}'))
with my_expander2:
    st.dataframe(new_data2)
    
comp_table

# print accuracy
st.warning(f'\nCurrent Accuracy Score: ' + str(round(acc_score*100,1)) + '%')
# except:
#     st.warning('Please pick a team to visualize the Stats for the Season')


# # To Download a File
# # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# # Heat Map 
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
