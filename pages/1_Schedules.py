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
import io

st.set_page_config(page_title="Schedules", page_icon="🗓")

st.title('NFL Football Schedules')

# st.markdown("Schedules")
# st.sidebar.header("Schedules")
st.sidebar.markdown("NFL Football Schedules")


st.markdown("""
This app performs simple webscraping of NFL Football Schedules
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")


#side bars
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))


# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2019/rushing.htm
@st.cache
def get_new_data_future(year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    
    - team : team name (str)
    - year : year (int)
    https://www.pro-football-reference.com/years/2022/games.htm
    '''
    # pull data
    url = f'https://www.pro-football-reference.com/years/{selected_year}/games.htm'
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
#     st.dataframe(df)

    # subset
    index = [0,1,2,3,4,5,6]#list(range(1,8))
    new_data = df.iloc[:,index].copy()
#     st.dataframe(new_data)
    
#     index = [0,1,4,8,9,10] + list(range(11,21))
#     new_data = df.iloc[:,index].copy()

#     rename columns
    col_names = [ 'Day', 'Date', 'Time', 'Winner/Tie', '', 'Loser/Tie', 'Boxscore']
    new_data.columns = col_names

    # upcoming dates
    new_data=new_data.loc[new_data['Boxscore']=='preview']

    # add week variable back
    week = list(range(1,len(new_data)+1))
    new_data.insert(0, 'Week', week)
    
#     # add team name
#     tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
#     new_data.insert(0, 'team_name', tn_col)

    # return a dataframe object
    if type(new_data) == pd.Series:
        new_data = new_data.to_frame().T
        return new_data.reset_index(drop=True)
    else:
        return new_data.reset_index(drop=True)

st.header(f'Upcoming Games Scheduled in {selected_year}')
new_data_future = get_new_data_future(year=selected_year)

# Filtering data
st.dataframe(new_data_future)



@st.cache
def get_new_data(year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    
    - team : team name (str)
    - year : year (int)
    https://www.pro-football-reference.com/years/2022/games.htm
    '''
    # pull data
    url = f'https://www.pro-football-reference.com/years/{selected_year}/games.htm'
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
#     index = [0,1,4,8,9,10] + list(range(11,21))
    new_data = df

    # rename columns
    col_names = [ 'Day', 'Date', 'Time', 'Winner/tie', '', 'Loser/tie', 'boxscore', 'PtsW', 'PtsL', 'YdsW', 'TOW', 'YdsL','TOL']
    new_data.columns = col_names

#     # encode results
#     result_encoder = {'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA},
#                      'TO_offense' : {'' : 0},
#                      'TO_defense' : {'' : 0}}
#     new_data.replace(result_encoder, inplace=True)

    # remove future dates
    new_data = new_data[new_data.PtsW.notnull()]

    # add week variable back
    week = list(range(1,len(new_data)+1))
    new_data.insert(0, 'Week', week)
    
#     # add team name
#     tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
#     new_data.insert(0, 'team_name', tn_col)

    # return a dataframe object
    if type(new_data) == pd.Series:
        new_data = new_data.to_frame().T
        return new_data.reset_index(drop=True)
    else:
        return new_data.reset_index(drop=True)

st.header(f'Games Scheduled in {selected_year}')
    
new_data = get_new_data(year=selected_year)

# Filtering data

my_expander = st.expander(label=(f'Click Here to display all Games Scheduled in {selected_year}'))
with my_expander:
    st.dataframe(new_data)
    st.text('Week: Week number in season')
    st.text('Time: Game Time, Eastern')
    st.text('PtsW: Points Scored by the winning team (first one listed)')
    st.text('PtsL: Points Scored by the losing team (second one listed)')
    st.text('YdsW: Yards Gained by the winning team (first one listed)')
    st.text('TOW: Turnovers by the winning team (first one listed)')
    st.text('YdsL: Yards Gained by the losing team (second one listed)')
    st.text('TOL: Turnovers by the losing team (second one listed)')
   
