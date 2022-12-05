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
