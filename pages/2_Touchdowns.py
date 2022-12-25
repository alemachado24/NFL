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
import datetime
import re
from bs4 import BeautifulStoneSoup

st.set_page_config(page_title="TD", page_icon="🏈")

st.title('NFL Touchdown Summary')

# st.markdown("Schedules")
# st.sidebar.header("Schedules")
st.sidebar.markdown("NFL Football Schedules")


st.markdown("""
This app performs simple webscraping of NFL Football Schedules
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")


symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']

selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

# Sidebar - Team selection
all_lower_sym = [x.lower() for x in symbols]
# st.text(all_lower_sym)
sorted_unique_team = sorted(all_lower_sym)
selected_team_full = st.multiselect('',team_names,default = team_names[11])


if selected_team_full[0] == 'Arizona Cardinals':
    short_name = 'CRD'
elif selected_team_full[0] == 'Atlanta Falcons':
    short_name = 'ATL'
elif selected_team_full[0] == 'Baltimore Ravens':
    short_name = 'RAV'
elif selected_team_full[0] == 'Buffalo Bills':
    short_name = 'BUF'
elif selected_team_full[0] == 'Carolina Panthers':
    short_name = 'CAR'
elif selected_team_full[0] == 'Chicago Bears':
    short_name = 'CHI'
elif selected_team_full[0] == 'Cincinnati Bengals':
    short_name = 'CIN'
elif selected_team_full[0] == 'Cleveland Browns':
    short_name = 'CLE'
elif selected_team_full[0] == 'Dallas Cowboys':
    short_name = 'DAL'
elif selected_team_full[0] == 'Denver Broncos':
    short_name = 'DEN'
elif selected_team_full[0] == 'Detroit Lions':
    short_name = 'DET'
elif selected_team_full[0] == 'Green Bay Packers':
    short_name = 'GNB'
elif selected_team_full[0] == 'Houston Texans':
    short_name = 'HTX'
elif selected_team_full[0] == 'Indianapolis Colts':
    short_name = 'CLT'
elif selected_team_full[0] == 'Jacksonville Jaguars':
    short_name = 'JAX'
elif selected_team_full[0] == 'Kansas City Chiefs':
    short_name = 'KAN'
elif selected_team_full[0] == 'Las Vegas Raiders':
    short_name = 'RAI'    
elif selected_team_full[0] == 'Los Angeles Chargers':
    short_name = 'SDG'
elif selected_team_full[0] == 'Los Angeles Rams':
    short_name = 'RAM'
elif selected_team_full[0] == 'Miami Dolphins':
    short_name = 'MIA'
elif selected_team_full[0] == 'Minnesota Vikings':
    short_name = 'MIN'
elif selected_team_full[0] == 'New England Patriots':
    short_name = 'NWE'
elif selected_team_full[0] == 'New Orleans Saints':
    short_name = 'NOR'
elif selected_team_full[0] == 'New York Giants':
    short_name = 'NYG'
elif selected_team_full[0] == 'New York Jets':
    short_name = 'NYJ'
elif selected_team_full[0] == 'Philadelphia Eagles':
    short_name = 'PHI'
elif selected_team_full[0] == 'Pittsburgh Steelers':
    short_name = 'PIT'
elif selected_team_full[0] == 'San Francisco 49ers':
    short_name = 'SFO'
elif selected_team_full[0] == 'Seattle Seahawks':
    short_name = 'SEA'
elif selected_team_full[0] == 'Tampa Bay Buccaneers':
    short_name = 'TAM'
elif selected_team_full[0] == 'Tennessee Titans':
    short_name = 'OTI'
elif selected_team_full[0] == 'Washington Football Team':
    short_name = 'WAS'

#         st.text(short_name)
if short_name == 'RAI':
    injury_name='LVR'
elif short_name == 'SDG':
    injury_name='LAC'
elif short_name == 'RAM':
    injury_name='LAR'
elif short_name == 'CRD':
    injury_name='ARI'
elif short_name == 'RAV':
    injury_name='BAL'
elif short_name == 'OTI':
    injury_name='TEN'
elif short_name == 'HTX':
    injury_name='HOU'
else:
    injury_name=short_name

st.header('Rushing and Receiving')
def get_scrimmage( year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    - team : team name (str)
    - year : year (int)
    '''
    # pull data
    url = f'https://www.pro-football-reference.com/years/{selected_year}/scrimmage.htm'
    html = requests.get(url).text
    #st.text(url) https://www.pro-football-reference.com/teams/gnb/2022.htm#passing

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    
    
########################################### Receiving ##########################################################    
    
    table_scri = soup.find('table', id='receiving_and_rushing') #rushing_and_receiving #all_advanced_rushing
#     st.dataframe(table_rush)
    
    tablerows_scri = table_scri.find_all('tr')[2:]
    
    data_scri = []

    for tablerow in tablerows_scri:#[2:]:
        try:
            data_scri.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])
        except:
            pass
#         

    df_scri = pd.DataFrame(data_scri)
#     st.dataframe(df_scri)
    
    index = [0,1] + [3] + [8] + [10] + [15] + [18,19] + [28] #+ list(range(7,21))
    df_scri_new = df_scri.iloc[:,index].copy()
    
    
    col_names = ['Player', 'Team', 'Position', 'Rec Yds', 'Rec TD','Catch%', 'Rush Yds', 'Rush TD', 'Rush & Rec TD']
    df_scri_new.columns = col_names
    
#     st.dataframe(df_scri_new)
    return df_scri_new
    

scri_td = get_scrimmage(year=selected_year)
# st.dataframe(scri_td)
scri_td_df = scri_td.loc[scri_td['Team']==injury_name]
st.dataframe(scri_td_df)


st.header('Passing')
def get_passing( year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    - team : team name (str)
    - year : year (int)
    '''
    # pull data
    url = f'https://www.pro-football-reference.com/years/{selected_year}/passing.htm'
    html = requests.get(url).text
    #st.text(url) https://www.pro-football-reference.com/teams/gnb/2022.htm#passing

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    
    
########################################### Passing ##########################################################    
    
    table_pass = soup.find('table', id='passing') #rushing_and_receiving #all_advanced_rushing
#     st.dataframe(table_rush)
    
    tablerows_pass = table_pass.find_all('tr')[1:]
#     st.dataframe(tablerows_rush)
#     st.write(tablerows_adv_pass[2:])
    
    data_pass = []

    for tablerow in tablerows_pass:#[2:]:
        try:
            data_pass.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])
        except:
            pass
#         

    df_pass = pd.DataFrame(data_pass)
    
    index = [0,1] + [3] + [10,11,12] #+ list(range(7,21))
    df_pass_new = df_pass.iloc[:,index].copy()
    
    
    col_names = ['Player', 'Team', 'Position', 'Yds Gained', 'Pass TD','TD%']
    df_pass_new.columns = col_names
    
#     st.dataframe(df_pass_new)
    return df_pass_new
    

pass_td = get_passing(year=selected_year)
pass_td_df = pass_td.loc[pass_td['Team']==injury_name]
st.dataframe(pass_td_df)


#MISSING TOUCHDOWN LOG


# st.header('TD Log')
# def get_td( year, team):
#     '''
#     Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
#     - team : team name (str)
#     - year : year (int)
#     '''
#     # pull data
#     url = f'https://www.pro-football-reference.com/teams/{short_name.lower()}/{selected_year}.htm'
#     html = requests.get(url).text
#     #st.text(url) https://www.pro-football-reference.com/teams/gnb/2022.htm#passing

#     # parse the data
#     soup = BeautifulSoup(html,'html.parser')
    
    
# ########################################### TD ##########################################################    
    
#     table_td = soup.find( id='team_td_log') #rushing_and_receiving #all_advanced_rushing
#     st.dataframe(table_td)
    
#     tablerows_td = table_td.find_all('tr')[1:]
# #     st.dataframe(tablerows_rush)
# #     st.write(tablerows_adv_pass[2:])
    
#     data_td = []

#     for tablerow in tablerows_td:#[2:]:
#         try:
#             data_td.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])
#         except:
#             pass
# #         

#     df_td = pd.DataFrame(data_td)
#     st.dataframe(df_td)
    

# log_td = get_td(year=selected_year, team=short_name.lower())
    


 
