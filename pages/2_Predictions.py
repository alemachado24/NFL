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


st.set_page_config(page_title="538 Forecast", page_icon="📈",layout="wide",)


st.title('Five Thirty Eight Forecast')

# 🎰

st.sidebar.markdown("NFL Football Forecast")


st.markdown("This app performs simple webscraping of NFL Football player stats data")
st.markdown("Data Sourcers: fivethirtyeight and pro-football-reference Websites")

#sidebar
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2023))))

general_stats, upcoming_games = st.tabs(["Standing Forecast", "Upcoming Games & Stats"])

##########################################
## Standing Forecast                    ##
##########################################

with general_stats:

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


##########################################
## Upcoming Games                    ##
##########################################

with upcoming_games:
    
    
    
    st.header("Upcoming Games")

    #---------------------------------Week Forecast & Upcomming Games
    row1_1, row1_2 = st.columns((3, 2))#st.columns(2)

    with row1_2:

        st.write(f'Games Win Probabilities in {selected_year} from FiveThirtyEight ')
        st.write('')
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

            soup = BeautifulSoup(html,'html.parser')

            #------------------------

            table2 = soup.find_all(class_=["h3","h4","tr"])


            data_tocheck2 = []

            for tablerow in table2:
                data_tocheck2.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('th')])
                data_tocheck2.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])


            df_tocheck2 = pd.DataFrame(data_tocheck2)

            col_names = ['Time', 'Team', 'Spread', 'Probability', 'Score']
            df_tocheck2.columns = col_names

            df_tocheck2 = df_tocheck2[df_tocheck2.Team.notnull()]

            df_tocheck2['Team'] = df_tocheck2['Team'].str.replace('Elo point spread','')
            df_tocheck2['Spread'] = df_tocheck2['Spread'].str.replace('Win prob.','')
            df_tocheck2['Probability'] = df_tocheck2['Probability'].str.replace('Score','')
            df_tocheck2['Score'] = df_tocheck2['Score'].fillna('')

            df_tocheck3=df_tocheck2.loc[df_tocheck2['Time']!='FINALEastern']
        #     st.dataframe(df_tocheck3)
            df_tocheck4=df_tocheck3.loc[df_tocheck3['Score']=='']

            return df_tocheck4


        st.dataframe(get_new_data538_games(selected_year))

    with row1_1:
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
    st.header('Pick a Team to Check its Stats')
    # try:
    # selected_team = st.multiselect('',sorted_unique_team,default = sorted_unique_team[11])
    selected_team_full = st.multiselect('',team_names,default = team_names[11])
    try:
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

        @st.cache
        def get_record(team, year):
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
            table = soup.find("strong", text="Record:")
            record=table.next_sibling.strip()
            comma = record.find(',')

            return record[:comma]  

        @st.cache
        def get_points_for(team, year):
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
            table = soup.find("strong", text="Points For:")
            pointsfor=table.next_sibling.strip()

            return pointsfor

        @st.cache
        def get_points_against(team, year):
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
            table = soup.find("strong", text="Points Against:")
            pointsagainst=table.next_sibling.strip()

            return pointsagainst
    #     get_record(team=short_name.lower(), year=selected_year)

        row1_5, row1_6,row1_7,row1_8 = st.columns((1.5, 3, 3, 3))#st.columns(2)





        #---------------------------------End of Select Team to Analyse        


        # st.text(selected_team_full)



        @st.cache
        def get_logo(team, year):
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
            table = soup.find("img",class_="teamlogo")
            logo = table['src']

            return logo

        with row1_5:
            st.image(get_logo(team=short_name.lower(), year=selected_year),caption=selected_team_full[0],width=150)


#---------------------------------This Year Home & Away
    
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
            #offense desense?
    #         st.dataframe(df)
            # subset (columns that I want)
            index = [0,1] + [4]+ list(range(7,21)) #+ list(range(11,21))
            new_data = df.iloc[:,index].copy()

            col_names = ['Day', 'Date', 'result', '@', 'Opponent','Team Score', 'Opponent Score','Off 1stD', 'Off TotY', 'Off PassY', 'Off RushY', 'Off TO','Def 1stD', 'Def TotY', 'Def PassY', 'Def RushY', 'Def TO']
            new_data.columns = col_names

            # rename columns
    #         col_names = ['Day', 'Date', 'result', '@', 'Opponent','Team Score', 'Opponent Score']
    #         new_data.columns = col_names

            # remove future dates
            new_data = new_data[new_data.result.notnull()]

            # add team name
            tn_col = pd.Series([f'{team.upper()}']).repeat(len(new_data)).reset_index(drop=True)
            new_data.insert(0, 'team', tn_col)

            # return a dataframe object
            if type(new_data) == pd.Series:
                new_data = new_data.to_frame().T
                return new_data.reset_index(drop=True)
            else:
                return new_data.reset_index(drop=True)
        new_data = get_new_data(team=short_name.lower(), year=selected_year)


        st.header(f"{selected_team_full[0]} games in {selected_year} season")
        new_data_22=new_data.loc[new_data['Opponent']!='Bye Week']
        result_encoder2 = {'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA}}
        new_data_22.replace(result_encoder2, inplace=True)
        new_data_2022=new_data_22.loc[new_data['Team Score']!='']
        chart_home_2022=new_data_2022.loc[new_data['@']=='']
        chart_away_2022=new_data_2022.loc[new_data['@']!='']
        
        with row1_7:
            st.header("Home and Away")
            st.markdown(f"Total games: {len(new_data_2022.index)}, total games won: {len(new_data_2022[(new_data_2022['result']==1)])}")
            st.markdown(f"Home games: {len(chart_home_2022.index)}, total games won: {len(chart_home_2022[(chart_home_2022['result']==1)])}")
#             st.text(len(chart_home_2022[(chart_home_2022['result']==1)]))
            st.markdown(f"Away games: {len(chart_away_2022.index)}, total games won: {len(chart_away_2022[(chart_away_2022['result']==1)])}")
        
        # st.dataframe(new_data_2022)

    #     st.write(f'Win & Losses for {selected_team_full[0]} when at Home & Away in {selected_year}')  

        row1_all_2022, row1_home_2022, row1_away_2022 = st.columns((6, 5,5))

        with row1_all_2022:
            inform = f"All Past Games this season: Win = 1, Loss/Tie=0"
            fig_all = px.line(new_data_2022, x="Date", y=new_data_2022['result'], title=inform)
            st.plotly_chart(fig_all, use_container_width=True)
            

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

        offanddef=new_data_22.loc[new_data_22['Day']!='']
        def_dataTO = offanddef.loc[offanddef['Def TO']!='']
        off_dataTO = offanddef.loc[offanddef['Off TO']!='']
        sumDefTO = def_dataTO['Def TO'].astype(int).sum(skipna = True) #sum(axis = 1, skipna = True)
        sumOffTO = off_dataTO['Off TO'].astype(int).sum(skipna = True)
        lenghtTO = len(offanddef.index)
    
    
        with row1_8:
            st.header("Offense & Defense")
            st.markdown(f"{selected_team_full[0]} had {sumDefTO} Defense TO in {lenghtTO} games")
            st.markdown(f"Average Defense TO in {selected_year}: {round(sumDefTO/lenghtTO,2)}")
            st.markdown(f"{selected_team_full[0]} had {sumOffTO} Offense TO in {lenghtTO} games")
            st.markdown(f"Average Offense TO in {selected_year}: {round(sumOffTO/lenghtTO,2)}")
 


    #----------------------Fantasy

    # st.header(f"Impact players from fantasy info for {selected_team_full[0]}")

        @st.cache
        def get_fantasy(year):
            '''
            Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).

            - team : team name (str)
            - year : year (int)
            https://www.pro-football-reference.com/years/2022/games.htm
            '''
            # pull data
            url = f'https://www.pro-football-reference.com/years/{selected_year}/fantasy.htm'
            html = requests.get(url).text
        #     st.text(url)

            # parse the data
            soup = BeautifulSoup(html,'html.parser')
            table = soup.find('table', id='fantasy')
            tablerows = table.find_all('tr')[2:]
            data = []

            for tablerow in tablerows:
                data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

            df = pd.DataFrame(data)


                # subset (columns that I want)
            index = [0,1,2] + list(range(25,32))#[7,8,9,10] #+ list(range(11,21))
            new_data = df.iloc[:,index].copy()
        #     st.dataframe(new_data)

        #     # rename columns
            col_names = ['Player', 'Team', 'FantPos', 'FantPt', 'PPR','DkPt', 'FdPt','VBD', 'PosRank', 'OvRan']
            new_data.columns = col_names
            new_data2=new_data[new_data.Player.notnull()]
        #     st.dataframe(new_data2)
    #         fanatasy_name = []
            if short_name == 'RAI':
                fanatasy_name='LVR'
            elif short_name == 'SDG':
                fanatasy_name='LAC'
            elif short_name == 'RAM':
                fanatasy_name='LAR'
            elif short_name == 'CRD':
                fanatasy_name='ARI'
            elif short_name == 'RAV':
                fanatasy_name='BAL'
            elif short_name == 'OTI':
                fanatasy_name='TEN'
            elif short_name == 'HTX':
                fanatasy_name='HOU'
            else:
                fanatasy_name=short_name


            new_data3 = new_data2.loc[new_data2['Team']==fanatasy_name]
        #     st.dataframe(new_data3)
            return new_data3

        
        #------------------End of Fantasy


        # st.header("Injury Reports")
        @st.cache
        def get_injuries():
            '''
            Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).

            - team : team name (str)
            - year : year (int)
            https://www.pro-football-reference.com/years/2022/games.htm
            '''
            # pull data
            url = f'https://www.pro-football-reference.com/players/injuries.htm'
            html = requests.get(url).text
        #     st.text(url)

            # parse the data
            soup = BeautifulSoup(html,'html.parser')
            table = soup.find('table', id='injuries')
            tablerows = table.find_all('tr')[1:]
            data = []
            data_names = []

            for tablerow in tablerows:
                data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

            for tablerow in tablerows:
                data_names.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('th')])

            df = pd.DataFrame(data)
            col_names = ['Team', 'Possition', 'Status', 'Injury','Practice Status']
            df.columns = col_names
            df2 = pd.DataFrame(data_names)
            col_names2 = ['Player']
            df2.columns = col_names2

            # to combined 2 dataframes by adding new columns
            combined_list = pd.concat([df2, df], axis=1)

            new_data = combined_list.loc[combined_list['Team']==short_name]
    #         st.dataframe(combined_list)
    #         st.text(short_name)
            return new_data
    
        st.header(f"Impact Players and Injury Report for {selected_team_full[0]}")
        offense_deffense = st.expander(label=f'Click Here for Impact Players and Injury Report')
        with offense_deffense:
            row1_3, row1_4 = st.columns((4, 4))#st.columns(2)
            
            with row1_3:
                # st.text(short_name)
                st.header(f"Impact players")
                st.dataframe(get_fantasy({selected_year}))         

            with row1_4:
                injuries = get_injuries()
                st.header("Injury Reports")
                st.dataframe(injuries)
                injury_count = len(injuries.index)
                

        with row1_6:
            st.header("Team Summary")
            st.markdown(f'Team record: {get_record(team=short_name.lower(), year=selected_year)}')
            st.markdown(f'Points For: {get_points_for(team=short_name.lower(), year=selected_year)}')
            st.markdown(f'Points Against: {get_points_against(team=short_name.lower(), year=selected_year)}')
            st.markdown(f'Team Injury Count: {injury_count}')
        
            #---------------------------------------Last Year Home & Away
        st.header(f'{selected_team_full[0]} games in {selected_year-1}')
        # try:
        @st.cache
        def get_last_data(team, year):
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
            index = [0,1] + [4]+ list(range(7,21))#[7,8,9,10] #+ list(range(11,21))
            new_data = df.iloc[:,index].copy()
    #         st.dataframe(new_data)

            # rename columns
            col_names = ['Day', 'Date', 'result', '@', 'Opponent','Team Score', 'Opponent Score','Off 1stD', 'Off TotY', 'Off PassY', 'Off RushY', 'Off TO','Def 1stD', 'Def TotY', 'Def PassY', 'Def RushY', 'Def TO']
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

        new_data = get_last_data(team=short_name.lower(), year=selected_year)

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
        
        my_expander_previous_0 = st.expander(label=f'Click Here to access Charts for {selected_year-1} Season for {selected_team_full[0]}')
        with my_expander_previous_0:

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

# #Influencing factors in games:


# QB is offense
# CB is defense corner back 
#-------------------

#Add the teams general overview Record, Points For and Points Against gives the ratingin the nfl 
#https://www.pro-football-reference.com/teams/htx/2022.htm

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
