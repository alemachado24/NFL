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
# from requests_html import HTMLSession
from datetime import date
import re

# st.set_page_config(page_title="538 Forecast", page_icon="📈")

st.title('Five Thirty Eight Forecast')

# 🎰

st.sidebar.markdown("NFL Football Forecast from 538")


st.markdown("""
This app performs simple webscraping of NFL Football player stats data & Predicted Wins Vs. Actual Wins
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn
* **Data source:** [https://projects.fivethirtyeight.com/](https://projects.fivethirtyeight.com/).
""")


#sidebar
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

st.header(f'Standing {selected_year} NFL Forecast from FiveThirtyEight ')
#------------- webscrap for elo
@st.cache
def get_new_data538(year):
    '''
    Function to pull NFL stats from 538 Reference (https://projects.fivethirtyeight.com/2022-nfl-predictions/).
    - year : year (int)
    '''
    # pull data
    url = f'https://projects.fivethirtyeight.com/{selected_year}-nfl-predictions/'
    html = requests.get(url).text
    #to make sure the url is corrext check the .text
#     st.text(url)

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
#     st.header('soup')

    #find the id in the table by inspecting the website
    table = soup.find("table", id="standings-table")
#     st.dataframe(table)

    #to find all row values in the table
    gdp_table_data = table.tbody.find_all("tr")[2:] 
#     st.text(gdp_table_data)

    #it's not for headings, it's for the row data
    headings = []
    for tablerow in gdp_table_data:
        # remove any newlines and extra spaces from left and right
        headings.append([tabledata.text.replace('\n', ' ').strip() for tabledata in tablerow.find_all("td")])
#     st.dataframe(headings)

    df = pd.DataFrame(headings)

    #Instead of dropping the columns and selecting the columns I'm going to use
    index = [0] + list(range(4,10))
    new_data_standings = df.iloc[:,index].copy()

    #Rename columns
    col_names = ['Elo Rating', 'Team', 'Division', 'Make to Playoffs', 'Win Division', '1st Round Bye', 'Win Super Bowl']
    new_data_standings.columns = col_names
    return new_data_standings
    
#Dataframe with Standing Predictions from 538
st.dataframe(get_new_data538(selected_year))
#--------------------

st.header(f'NFL Game Forecast in {selected_year} from FiveThirtyEight ')
#------------- webscrap for elo
@st.cache
def get_new_data538_games(year):
    '''
    Function to pull NFL stats from 538 Reference (https://projects.fivethirtyeight.com/2022-nfl-predictions/).
    - year : year (int)
    '''
    # pull data
    url = f'https://projects.fivethirtyeight.com/{selected_year}-nfl-predictions/games/'
    html = requests.get(url).text
    #to make sure the url is corrext check the .text
#     st.text(url)

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    #---------------
    #Way # 2 to pull the data from html
#     s = HTMLSession()
#     r=s.get(url)
#     read html in an easy way:
#     tabledata = pd.read_html(url,attrs={'id': 'week-group'})
#     st.write(tabledata)
#     st.write(tableheader)
    #------------------------

    body_web = soup.find('body')
#     st.text(body_web)
#     select today
#     st.text(date.today())

#   este me da el numero de la semana
#     week_number = body_web.find_all("h3", {"class": "h3"})
#     st.text(week_number[2])
#     st.text(len(week_number))
    
#   este me da el dia del juego
#     week_header = body_web.find_all("h4", {"class": "h4"})
#     st.text(len(week_header))

#   este me da el nombre del equipo con regrex
#     week_teams = body_web.select('td[class*="td text team"]')
#     st.text(len(week_teams))
    
#   este me da la probabilidad de ganar
#     week_chance = body_web.find_all("td", {"class": "td number chance"})
#     st.text(len(week_chance))

    #to find a column with all consecutive values
    trial=(body_web.find_all(class_=["h3","h4",re.compile("td text team"),"td number chance"]))
    
#     st.write(trial[:391])#[1:5]
    trial2 = pd.DataFrame(trial)
    return (trial2[:391])
   
    
#Dataframe with Standing Predictions from 538
st.dataframe(get_new_data538_games(selected_year))
#----------------------------


#comment out from here, un comment later:


# # load dataset
# # Make sure the url is the raw version of the file on GitHub
# url_csv = "https://raw.githubusercontent.com/alemachado24/NFL/main/season_2021.csv" 
# download = requests.get(url_csv).content
# nfl = pd.read_csv(io.StringIO(download.decode('utf-8')),sep=',')

# #If you have the file saved in your computer
# # nfl = pd.read_csv('/Users/am/Desktop/AleClasses/NFL/season_2021.csv')

# # inspect first few rows
# # nfl.head()

# #Title + Table in Display
# st.header('-----------------------Display Player Stats for 2021 Season')
# # st.dataframe(nfl)

# #side bars
# st.sidebar.header('User Input Features')

# graph_elements = ['1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense',
#        'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense',
#        'RushY_defense', 'TO_defense']

# # Sidebar - Graph Elements
# selected_stat = st.sidebar.multiselect('Stats', graph_elements,default = graph_elements[4])

# st.header('Visualize the stats for 2021 Season by Loss, Tie & Win')
# # change stat to view plot
# try:
#     stat = selected_stat[0] #'1stD_offense'
#     st.text(selected_stat[0])

#     # box plot of stat
#     stat_plot = sns.boxplot(x='result', y=stat, data=nfl)
#     #This turn off an alarm for using a Global Variable
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     # plot labels
#     # stat_plot.set_xticklabels(['loss/tie','win'])
#     st.pyplot(x='result', y=stat, data=nfl)
# #     plt.ioff()

# except:
#     st.warning('Please pick a stat to visualize the plot')

# # To Transform Wins Ties And L to numeric numbers
# # nested dictionary to encode alphanumeric values to numeric values
# result_encoder = {'result': {'W': 1, 'T': 0, 'L': 0}}
# # encode result column using encoder
# nfl.replace(result_encoder, inplace=True)
# # check result value counts (You should have numbers instead of W, T or L
# # nfl.result.value_counts()

# st.header('Visualize the stats for 2021 Season by Loss/Tie & Win')
# # change stat to view plot
# try:
#     stat = selected_stat[0] #'1stD_offense'
#     st.text(selected_stat[0])

#     # box plot of stat
#     stat_plot = sns.boxplot(x='result', y=stat, data=nfl)
#     #This turn off an alarm for using a Global Variable
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     # plot labels
#     stat_plot.set_xticklabels(['Loss/Tie','Win'])
#     st.pyplot(x='result', y=stat, data=nfl)
# #     plt.ioff()
# except:
#     st.warning('Please pick a stat to visualize the plot')

# # # list feature names (columns in the NFL File)
# # st.text(nfl.columns[8:])

# #Header to check Model Accuracy
# st.header('Model Accuracy')

# # select feature variables to be scaled
# features = nfl.iloc[:,8:]
# scaler = StandardScaler()

# # st.dataframe(features)

# # fit the transformer to the features
# scaler.fit(features)

# # transform and save as X
# X = scaler.transform(features)

# # save result variable as y
# y = nfl['result']

# # create train-test split of the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# # create the classifier
# lrc = LogisticRegression()

# # fit classifier to the training data
# lrc.fit(X_train, y_train)

# # predict with the classifier using the .predict() function
# y_pred = lrc.predict(X_test)

# # view the model accuracy with the accuracy_score() function
# accuracy_score(y_pred,y_test)

# # create a list of penalties
# penalties = ['l1', 'l2']
# # create a list of values for C
# C = [0.01, 0.1, 1.0, 10.0, 1000.0]

# #Check best values for Penalty and C values in the regression
# for penalty in penalties:
#     for c in C:

#         # instantiate the classifier
#         lrc_tuned = LogisticRegression(penalty=penalty, C=c, solver='liblinear')

#         # fit the classifier to the training data
#         lrc_tuned.fit(X_train, y_train)
        
#         # predict with the classifier using the .predict() function
#         y_pred = lrc_tuned.predict(X_test)

#         # view the model accuracy with the accuracy_score() function
#         accuracy = accuracy_score(y_test, y_pred)
#         accuracy_rd = round(accuracy*100,1)
        
#         # print accuracy for each combination of penalty and C
        
# #         st.text(f'Accuracy: {accuracy_rd}% | penalty = {penalty}, C = {c}')
        
# #Missing to print the max accuracy instead of all of them !!!

# #         st.text(f'Accuracy: {max(accuracy_rd)}% | penalty = {penalty}, C = {c}')
# # st.text('Optimal Penalty & C Value:')
# # max_value = None
# # for n in accuracy_score(y_test, y_pred):
# #     if max_value is None or n > max_value: max_value = n
# # st.text(f'Accuracy: {max_value}%')


# #Missing to set this numbers to a variable dependable from above same for test sizes!!!
# # optimal penalty and C
# penalty = 'l1'
# C = 0.1

# #Missing to print the optimal test size instead of all of them !!!

# # create a list of test_sizes
# test_sizes = [val/100 for val in range(20,36)]

# for test_size in test_sizes:

#     # train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#     # instantiate the classifier
#     lrc_tts = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

#     # fit the classifier to the training data
#     lrc_tts.fit(X_train, y_train)

#     # predict with the classifier using the .predict() function
#     y_pred = lrc_tts.predict(X_test)

#     # view the model accuracy with the accuracy_score() function
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_rd = round(accuracy*100,1)
    
#     # print accuracy for each combination of penalty and test size
#     st.text(f'Accuracy: {accuracy_rd}% | test size = {test_size}')
    

# #Missing to set this numbers to a variable dependable from above same for test sizes!!!
# # set the test size and hyperparameters
# test_size = 0.25
# penalty = 11
# C = 0.1

# # train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # instantiate the classifier
# optLr = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

# # fit the classifier to the training data
# lrc.fit(X_train, y_train)


# # get importance
# importance = abs(lrc.fit(X_train, y_train).coef_[0])

# # visualize feature importance
# # sns.barplot(x=features.columns, y=importance)

# #Header + Table for Importance
# st.header('Feature Importance for Logistic Regression for 2021 Season')
# chart_data = pd.DataFrame(importance,features.columns)
# # st.bar_chart(chart_data)


# # summarize feature importance
# for i,v in enumerate(importance.round(2)):
#     st.text(f'Feature: {features.columns[i]}, Score: {v}')


# # Web scraping of NFL player stats
# # https://www.pro-football-reference.com/years/2019/rushing.htm
# symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
# team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']

# # Sidebar - Team selection
# all_lower_sym = [x.lower() for x in symbols]
# # st.text(all_lower_sym)
# sorted_unique_team = sorted(all_lower_sym)
# selected_team = st.sidebar.multiselect('Team', sorted_unique_team,default = all_lower_sym[11])

# # try:
# @st.cache
# def get_new_data(team, year):
#     '''
#     Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).

#     - team : team name (str)
#     - year : year (int)
#     '''
#     # pull data
#     url = f'https://www.pro-football-reference.com/teams/{selected_team[0]}/{selected_year}.htm'
#     html = requests.get(url).text
#     #st.text(url)

#     # parse the data
#     soup = BeautifulSoup(html,'html.parser')
# #     st.text(soup)
#     table = soup.find('table', id='games')
# #     st.header('table')
# #     st.text(table)
#     tablerows = table.find_all('tr')[2:]
#     data = []

#     for tablerow in tablerows:
#         data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

#     df = pd.DataFrame(data)


#     # subset
#     index = [0,1,4,8,9,10] + list(range(11,21))
#     new_data = df.iloc[:,index].copy()

#     # rename columns
#     col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score', '1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense', 'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense', 'RushY_defense', 'TO_defense']
#     new_data.columns = col_names

#     # encode results
#     result_encoder = {'result': {'L': 0, 'T': 0,'W': 1,'' : 'Not played yet'#pd.NA
#                                 }}#,
# #                      'TO_offense' : {'' : pd.NA},
# #                      'TO_defense' : {'' : pd.NA}}
#     new_data.replace(result_encoder, inplace=True)
#     new_data=new_data.loc[new_data['opponent']!='Bye Week']
# #     st.text(date.today())
#     st.dataframe(new_data)

#     #Missing part to fill nulls with 0 for future predictions
#     # remove future dates
# #     new_data = new_data[new_data.result.notnull()]


#     # add week variable back
#     week = list(range(1,len(new_data)+1))
#     new_data.insert(0, 'week', week)

#     # add team name
#     tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
#     new_data.insert(0, 'team_name', tn_col)

#     # return a dataframe object
#     if type(new_data) == pd.Series:
#         new_data3 = new_data.to_frame().T
#         return new_data.reset_index(drop=True)
#     else:
#         return new_data.reset_index(drop=True)


# st.header(f'Display Player Stats of {selected_team[0].upper()} in {selected_year}')
# new_data = get_new_data(team=selected_team[0].upper(), year=selected_year)
# st.dataframe(new_data)

# pred_games_df = new_data.loc[new_data['result']=='Not played yet']

# #------------------------------------- do I need this? Probably yes?
# new_data_played = new_data.loc[new_data['result']!='Not played yet']
# result_encoder = {
#                      'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA},
#                      'TO_offense' : {'' : 0},
#                      'TO_defense' : {'' : 0}}
# new_data_played.replace(result_encoder, inplace=True)
# st.dataframe(new_data_played)
# #--------------------------------------------

# # select just the game stats
# new_X = new_data_played.loc[:,features.columns]

# # standardize using original data's scaling
# new_X_sc = scaler.transform(new_X)

# # get new predictions
# new_preds = lrc.fit(X_train, y_train).predict(new_X_sc)

# #FROM NEW WEBSITE
# # new_preds = new_preds[:,1]

# # st.dataframe(new_data_played)


# #-------------------- to display the text with probablities
# # def display(new_preds,new_data):
# #     for g in range(len(new_preds)):
# #         win_prob = round(new_preds[g],2)
# #         away_team = new_data.loc[g,'team_name']
# #         home_team = new_data.loc[g,'opponent']
# #         st.text(f'The {away_team} have a probability of {win_prob} of beating the {home_team}.') 
# #         st.text('test')
# # st.dataframe(new_preds)
# # st.dataframe(new_data_played)
# # display(new_preds,new_data_played)
# #----------------------

# #predicted win or lose/tie
# st.header('new_preds')
# st.dataframe(new_preds)
# # st.dataframe(new_data_played)


# # get actual results and set type to float
# # new_results = new_data_played['result'].astype(float)

# st.dataframe(new_data_played['result'])

# # get accuracy score for new data
# # acc_score = accuracy_score(new_results, new_preds)

# # select only game data
# col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
# game_data = new_data_played.loc[:,col_names]

# st.dataframe(game_data)

# # create comparison table
# comp_table = game_data.assign(predicted = new_preds, actual = new_data_played['result'].astype(int))

# # st.header('Predicted Wins vs Actual Wins')
# # print title and table
# st.header(f'Predicted Wins vs Actual Wins for {selected_team[0].upper()} in {selected_year}')
# comp_table

# # print accuracy
# # st.text(f'\nCurrent Accuracy Score: ' + str(round(acc_score*100,1)) + '%')

# col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
# pred_games_df2 = pred_games_df.loc[:,features.columns]

# # st.text(new_results)

# st.dataframe(y_pred)
# st.dataframe(pred_games_df2)

# X_test_new = pred_games_df2
# y_pred = lrc.predict(X_test)
# # y_pred = lrc.predict(X_test_new)
# y_pred = y_pred[:,1]

# st.dataframe(y_pred)
# st.dataframe(pred_games_df2)



# # display(y_pred,pred_games_df)

# # except:
# #     st.warning('Please pick a team to visualize the Stats for the Season')

    
# #
# # def display(y_pred,X_test):
# #     for g in range(len(y_pred)):
# #         win_prob = round(y_pred[g],2)
# #         away_team = X_test.reset_index().drop(columns = 'index').loc[g,'away_name']
# #         home_team = X_test.reset_index().drop(columns = 'index').loc[g,'home_name']
# #         print(f'The {away_team} have a probability of {win_prob} of beating the {home_team}.') 


# # # To Download a File
# # # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# # def filedownload(df):
# #     csv = df.to_csv(index=False)
# #     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
# #     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
# #     return href

# # st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# # # Heat Map 
# # if st.button('Intercorrelation Heatmap'):
# #     st.header('Intercorrelation Matrix Heatmap')
# #     df_selected_team.to_csv('output.csv',index=False)
# #     df = pd.read_csv('output.csv')

# #     corr = df.corr()
# #     mask = np.zeros_like(corr)
# #     mask[np.triu_indices_from(mask)] = True
# #     with sns.axes_style("white"):
# #         f, ax = plt.subplots(figsize=(7, 5))
# #         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
# #     st.pyplot()