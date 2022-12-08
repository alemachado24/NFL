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
from datetime import date
import re
import datetime
import plotly.express as px
# import pages/1_Schedules as sc


st.set_page_config(page_title="538 Forecast", page_icon="📈")


st.title('Five Thirty Eight Forecast')

# 🎰

st.sidebar.markdown("NFL Football Forecast")


st.markdown("""
This app performs simple webscraping of NFL Football player stats data & Predicted Wins Vs. Actual Wins
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, requests, bs4, sklearn
* **Data source:** [https://projects.fivethirtyeight.com/](https://projects.fivethirtyeight.com/).
""")

#sidebar
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

my_date = datetime.date.today()  
year, week_num, day_of_week = my_date.isocalendar()
# st.text(week_num-36)
current_day=day_of_week

if current_day == 1:
    weeks_current=int(week_num-36)
else:
    weeks_current=int(week_num-35)

weeks_to_select = ('Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8', 'Week 9', 'Week 10', 'Week 11', 'Week 12', 'Week 13', 'Week 14', 'Week 15', 'Week 16', 'Week 17', 'Week 18')

default_index=weeks_to_select[weeks_current]
selected_week = st.sidebar.selectbox('Week', weeks_to_select)#,index=weeks_to_select(int(weeks_current)))#default_index)

previous_week='Week '+ str(weeks_to_select.index(selected_week))

after_week='Week '+ str(weeks_to_select.index(selected_week)+2)

if current_day == 1:
    weeks_toexclude='Week '+ str(week_num-37)
else:
    weeks_toexclude='Week '+ str(week_num-36)


st.header(f'Standing {selected_year} NFL Forecast from FiveThirtyEight ')
#---------------------------------538 Prediction Table
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
#---------------------------------End Of 538 Prediction Table

st.header("Upcoming Games - Forecasted")

#---------------------------------Week Forecast & Upcomming Games
row1_1, row1_2 = st.columns((2, 4))#st.columns(2)

with row1_1:

    st.write(f'NFL Game Forecast in {selected_year} from FiveThirtyEight ')
    #------------- webscrap for elo
    @st.cache(hash_funcs={pd.DataFrame: lambda _: None})
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
        week_teams = body_web.select('td[class*="td text team"]')
    #     st.dataframe((week_teams))

    #   este me da la probabilidad de ganar
    #     week_chance = body_web.find_all("td", {"class": "td number chance"})
    #     st.text(len(week_chance))

        #to find a column with all consecutive values
        trial=(body_web.find_all(class_=["h3","h4",re.compile("td text team"),"td number chance"])) #,"th time",not("timezone")
    #     st.dataframe((trial))
        initial_count_to_exclude=0
        for x in trial:
    #         st.text(x)
            initial_count_to_exclude = initial_count_to_exclude + 1
            if x.text == weeks_toexclude:
                weeks_out = initial_count_to_exclude

        trial2 = pd.DataFrame(trial)
        initial_count=0
        for x in trial:
    #         st.text(x)
            initial_count = initial_count + 1
            if x.text == previous_week:
                week_before = initial_count
            if x.text == selected_week:
                weeks = initial_count
            if x.text == after_week:
                weeks_after = initial_count
            if x.text == weeks_toexclude:
                weeks_exclude = initial_count

        if selected_week == 'Week 1':
            return (trial2[weeks-1:])
        elif selected_week == 'Week 18':
            return (trial2[weeks-1:weeks_exclude-1])
        elif selected_week[-2:] == weeks_toexclude[-2:]:
            return (trial2[weeks-1:week_before-1])
        elif selected_week[-2:] < weeks_toexclude[-2:]:
            return (trial2[weeks-1:week_before-1])
        else:
            return (trial2[weeks-1:weeks_after-1])

    st.dataframe(get_new_data538_games(selected_year))

with row1_2:
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
        tablerows = table.find_all('tr')[1:]
        data = []

        for tablerow in tablerows:
            data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

        df = pd.DataFrame(data)
    #     st.dataframe(df)

        # subset
        index = [0,1,2,3,4,5,6]
        new_data = df.iloc[:,index].copy()

    #     rename columns
        col_names = [ 'Day', 'Date', 'Time', 'Winner/Tie', '', 'Loser/Tie', 'Boxscore']
        new_data.columns = col_names

        # upcoming dates
        new_data=new_data.loc[new_data['Boxscore']=='preview']

        # add week variable back
#         week = list(range(1,len(new_data)+1))
#         new_data.insert(0, 'Week', week)

        # return a dataframe object
        if type(new_data) == pd.Series:
            new_data = new_data.to_frame().T
            return new_data.reset_index(drop=True)
        else:
            return new_data.reset_index(drop=True)

    st.write(f'Upcoming Games Scheduled in {selected_year}')
    st.write('')
    new_data_future = get_new_data_future(year=selected_year)

    # Filtering data
    st.dataframe(new_data_future)

#---------------------------------End of Week Forecast & Upcomming Games

#---------------------------------Select Team to Analyse
# Web scraping of NFL player stats
# https://www.pro-football-reference.com/years/2019/rushing.htm
symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']

# Sidebar - Team selection
all_lower_sym = [x.lower() for x in symbols]
# st.text(all_lower_sym)
sorted_unique_team = sorted(all_lower_sym)
st.header('Pick A Team To See Stats')
try:
    # selected_team = st.multiselect('',sorted_unique_team,default = sorted_unique_team[11])
    selected_team_full = st.multiselect('',team_names,default = team_names[11])
    # st.text(selected_team_full)
    #---------------------------------End of Select Team to Analyse
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

    # index=selected_team.index(selected_team[0])
    # st.text(index)
    # st.text(short_name.lower())

    #---------------------------------This Year Home & Away
    # try:
    @st.cache
    def get_new_data(team, year):
        '''
        Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
        - team : team name (str)
        - year : year (int)
        '''
        # pull data
        url = f'https://www.pro-football-reference.com/teams/{short_name.lower()}/{selected_year}.htm'
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
        index = [0,1] + [4]+ [7,8,9,10] #+ list(range(11,21))
        new_data = df.iloc[:,index].copy()

        # rename columns
        col_names = ['Day', 'Date', 'result', '@', 'Opponent','Team Score', 'Opponent Score']
        new_data.columns = col_names

        # remove future dates
        new_data = new_data[new_data.result.notnull()]

        # add team name
        tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
        new_data.insert(0, 'team', tn_col)

        # return a dataframe object
        if type(new_data) == pd.Series:
            new_data = new_data.to_frame().T
            return new_data.reset_index(drop=True)
        else:
            return new_data.reset_index(drop=True)
    new_data = get_new_data(team=short_name.lower(), year=selected_year)

    st.header(f"{selected_team_full[0]} past games this season")
    new_data_22=new_data.loc[new_data['Opponent']!='Bye Week']
    new_data_2022=new_data_22.loc[new_data['Team Score']!='']
    chart_home_2022=new_data_2022.loc[new_data['@']=='']
    chart_away_2022=new_data_2022.loc[new_data['@']!='']
    # st.dataframe(new_data_2022)

    st.write(f'Win & Losses for {selected_team_full[0]} when at Home & Away in {selected_year}')

    row1_home_2022, row1_away_2022 = st.columns(2)

    with row1_home_2022:
        inform = f"Home Games: Win = 1, Loss/Tie=0"
        fig_home = px.line(chart_home_2022, x="Date", y=chart_home_2022['result'], title=inform)
        st.plotly_chart(fig_home, use_container_width=True)

    with row1_away_2022:
        inform = f"Away Games: Win = 1, Loss/Tie=0"
        fig_away = px.line(chart_away_2022, x="Date", y=chart_away_2022['result'], title=inform)
        st.plotly_chart(fig_away, use_container_width=True)


    my_expander_lastseason = st.expander(label=f'Click Here to access More Stats for {selected_year} Season for {selected_team_full[0]}')
    with my_expander_lastseason:
        st.dataframe(new_data_22)
    #---------------------------------------End Of This Year Home & Away


    #---------------------------------------Last Year Home & Away
    st.header(f'{selected_team_full[0]} games last season ({selected_year-1})')
    # try:
    @st.cache
    def get_new_data(team, year):
        '''
        Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).

        - team : team name (str)
        - year : year (int)
        '''
        # pull data
        url = f'https://www.pro-football-reference.com/teams/{short_name.lower()}/{selected_year-1}.htm'
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
    #     st.dataframe(df)

        # subset (columns that I want)
        index = [0,1] + [4]+ [7,8,9,10] #+ list(range(11,21))
        new_data = df.iloc[:,index].copy()
    #     st.dataframe(new_data)

        # rename columns
        col_names = ['Day', 'Date', 'result', '@', 'Opponent','Team Score', 'Opponent Score']
        new_data.columns = col_names

        # remove future dates
        new_data = new_data[new_data.result.notnull()]

        # add week variable back
    #     week = list(range(1,len(new_data)+1))
    #     new_data.insert(0, 'week', week)

        # add team name
        tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
        new_data.insert(0, 'team', tn_col)

        # return a dataframe object
        if type(new_data) == pd.Series:
            new_data = new_data.to_frame().T
            return new_data.reset_index(drop=True)
        else:
            return new_data.reset_index(drop=True)

    new_data = get_new_data(team=short_name.lower(), year=selected_year)

    index = [2]+ [3,4] +[6,7] #+ list(range(11,21))
    chart_previous_season = new_data.iloc[:,index].copy()
    # encode results
    result_encoder = {'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA}}
    chart_previous_season.replace(result_encoder, inplace=True)

    col_names = ['Date', 'Result', '@','Team Score', 'Opponent Score']
    chart_previous_season.columns = col_names

    chart_previous_season=chart_previous_season.loc[new_data['Team Score']!='']

    chart_home_previous=chart_previous_season.loc[new_data['@']=='']
    chart_away_previous=chart_previous_season.loc[new_data['@']!='']

    # st.dataframe(chart_home_previous)
    # st.dataframe(chart_away_previous)

    st.write(f'Win & Losses for {selected_team_full[0]} when at Home & Away in {selected_year-1}')

    row1_home_previous, row1_away_previous = st.columns(2)#st.columns(2)

    with row1_home_previous:

    #     st.write("Home Games")
        inform = f"Home Games: Win = 1, Loss/Tie=0"
        fig_home_previous = px.line(chart_home_previous, x="Date", y=chart_home_previous['Result'], title=inform)
        st.plotly_chart(fig_home_previous, use_container_width=True)

    with row1_away_previous:

    #     st.write("Away Games")
        inform = f"Away Games: Win = 1, Loss/Tie=0"
        fig_away_away = px.line(chart_away_previous, x="Date", y=chart_away_previous['Result'], title=inform)
        st.plotly_chart(fig_away_away, use_container_width=True)


    my_expander_previous = st.expander(label=f'Click Here to access More Stats for {selected_year-1} Season for {selected_team_full[0]}')
    with my_expander_previous:
        st.dataframe(new_data)
    #---------------------------------------End of Last Year Home & Away
except:
    st.warning('Please select a team')



st.header("Schedules filtered by new team to select")

st.text("Offense & Deffense from: Schedule & Game Results https://www.pro-football-reference.com/teams/gnb/2022.htm")

st.text("Injury report by team https://www.pro-football-reference.com/teams/gnb/2022_injuries.htm")

st.text("Impact players from fantasy info? https://www.pro-football-reference.com/years/2022/fantasy.htm")
# #Influencing factors in games:

#this could be from data from schedules -------------
# Whos home and whos away PRO FOOTBALL DEFINED HOME BY @

# Win streak at home? SCRAPE FOR PREVIOUS DATA
# How good the defense is for both teams
# How good the offense is for both teams

# QB is offense
# CB is defense corner back 
#-------------------

# Injuries to the starters for both teams on offense and defense
#what to look for in injuries? PRO FOOTBALL
# ACL and Achilles are the worst usually means a player is done for the season, anything broken is a problem 

# How good the impact players have been playing 
# Impact player by your standard would be the players with the highest numbers that affect the game…. Rushing yards, passing yards, touchdowns, YPC for a RB, YPG for a WR… QBR, for the season for a QB… sacks per game for a DL, LB, DB, interceptions for the season for a DB 

# Can the other team manage to score a ton of points if it becomes a shootout
# whats a shootout
#Shootout: breaking a tie score at the end of overtime in which five players from each team alternate shooting at the opponent's goal

# coaching staff for both teams
# have the coaches played against before? who won?

#----------------------------
