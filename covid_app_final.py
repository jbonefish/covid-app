# imports
import pandas as pd
import numpy as np
import csv
import streamlit as st
import datetime as dt
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import math
import random

import certifi
import ssl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# for county level chlorpleth plotly.express maps
import json

from urllib.request import urlopen
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json', context=ssl.create_default_context(cafile=certifi.where())) as response:
    counties = json.load(response)
#import urllib.request as urlrq
#response = urlrq.urlopen('https://github.com/plotly/datasets/blob/master/geojson-counties-fips.json', context=ssl.create_default_context(cafile=certifi.where()))
#counties = json.load(response)

# reference
folder = '~/Documents/county_data/'

covid_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
geo_url = "https://www.latlong.net/category/states-236-14.html"
geo_file = 'Geocodes_USA_with_Counties.csv'
census_files = ['HEA02', 'RHI01', 'LND01', 'POP01']
metro_file = 'Revised_core_based_statistical_area_for_the_US__Sept__2018.csv'

# groups from census bureau
groups = pd.read_excel( 'Mastgroups.xls')
subgroups = pd.read_excel( 'Mastdata.xls')

# make a dict of columns names and descriptions
col_name_dict = pd.Series(subgroups['Item_Description'].values,index=subgroups['Item_Id']).to_dict()

from math import radians, cos, sin, asin, sqrt
def dist(lat1, long1, lat2, long2):
    """
Replicating the same formula as mentioned in Wiki
    """
    # convert decimal degrees to radians
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

# function to make primary dataframe and cache it
@st.cache
def get_data(covid_url, geo_url, geo_file,list_of_files):
    r = {'state': ["California", "Nevada", "Illinois", "Indiana", "Michigan", "Minnesota", "Ohio",
                   "Wisconsin","Delaware", "District of Columbia", "Kentucky", "Maryland", "North Carolina",
                  "Tennessee", "Virginia", "West Virginia","Iowa","Kansas","Missouri","Nebraska","North Dakota",
                  "South Dakota","Arizona", "Colorado", "Idaho", "Montana", "New Mexico", "Utah", "Wyoming",
                  "Connecticut", "Maine","Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey",
                   "New York","Pennsylvania","Oregon", "Washington","Arkansas","Louisiana","Oklahoma","Texas",
                   "Alabama", "Florida","Georgia","Mississippi","South Carolina","Hawaii","Alaska","Puerto Rico"]
                  ,
         'region': ["Far_West", "Far_West", "Great_Lakes","Great_Lakes","Great_Lakes","Great_Lakes",
                    "Great_Lakes","Great_Lakes","Midsouth","Midsouth","Midsouth","Midsouth","Midsouth",
                   "Midsouth","Midsouth","Midsouth","Midwest","Midwest","Midwest","Midwest","Midwest","Midwest",
                   "Mountain_West","Mountain_West","Mountain_West","Mountain_West","Mountain_West","Mountain_West",
                   "Mountain_West","New England","New England","New England","New England","New England","New England",
                    "Northeast","Northeast","Northeast","Northwest","Northwest","South_Central","South_Central",
                    "South_Central","South_Central","Southeast","Southeast","Southeast","Southeast","Southeast",
                    "non_contiguous","non_contiguous","non_contiguous"],

        'state_abbr': ["CA", "NV", "IL", "IN", "MI", "MN", "OH", "WI","DE", "DC", "KY", "MD", "NC",
                  "TN", "VA", "WV","IA","KS","MO","NE","ND", "SD","AZ", "CO", "ID", "MT", "NM", "UT", "WY",
                  "CT", "ME","MA", "NH", "RI", "VT", "NJ", "NY","PA","OR", "WA","AR","LA","OK","TX",
                   "AL", "FL","GA","MS","SC","HI","AK","PR"]
                   }

    regions = pd.DataFrame(data=r)

    Far_West_list = ["California", "Nevada"]
    Great_Lakes_list = ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"]
    Midsouth_list = ["Delaware", "District of Columbia", "Kentucky", "Maryland", "North Carolina","Tennessee", "Virginia", "West Virginia"]
    Midwest_list = ["Iowa","Kansas","Missouri","Nebraska","North Dakota", "South Dakota"]
    Mountain_West_list = ["Arizona", "Colorado", "Idaho", "Montana", "New Mexico", "Utah", "Wyoming"]
    New_England_list = ["Connecticut", "Maine","Massachusetts", "New Hampshire", "Rhode Island", "Vermont"]
    Northeast_list = ["New Jersey","New York","Pennsylvania"]
    Northwest_list = ["Oregon", "Washington"]
    South_Central_list = ["Arkansas","Louisiana","Oklahoma","Texas",]
    Southeast_list = ["Alabama", "Florida","Georgia","Mississippi","South Carolina"]
    non_contiguous_list = ["Hawaii","Alaska","Puerto Rico"]

    covid = pd.read_csv(covid_url, dtype={'fips':str})
    covid['county'] = covid['county'].str.lower()

    df = pd.merge(covid,regions,on='state')
    df['date'] = pd.to_datetime(df['date'])

    lat_long = pd.read_html(geo_url)
    lat_long = lat_long[0]

    f = lambda x: x["Place Name"].split(",")

    lat_long["state"] = lat_long.apply(f, axis=1)
    lat_long['state'] = lat_long['state'].apply(lambda x: x[0])
    lat_long['state'].replace({'Missouri State': "Missouri", "Washington State":"Washington"}, inplace=True)
    lat_long.rename(columns={'Latitude':'state_lat','Longitude':'state_lon'}, inplace=True)
    dc_row = {'Place Name':'District of Columbia, the USA', 'state_lat':38.9, 'state_lon':-77, 'state':'District of Columbia'}
    #append row to the dataframe
    lat_long = lat_long.append(dc_row, ignore_index=True)

    regions = pd.merge(regions,lat_long,on='state')
    regions.drop(columns=['Place Name'], inplace=True)
    regions_latlong = regions.groupby('region')['state_lat','state_lon'].median()
    regions_latlong.rename(columns={'state_lat':'region_lat','state_lon':'region_lon'}, inplace=True)
    regions_latlong.reset_index(inplace=True)

    county_latlong = pd.read_csv( geo_file, dtype={'zip':'str'})
    county_latlong.drop(columns=['type', 'world_region','country','decommissioned','estimated_population','notes'], inplace=True)
    county_latlong.rename(columns={'state':'state_abbr', 'latitude':'county_lat','longitude':'county_lon'}, inplace=True)
    county_latlong['county'].replace({'Jefferson Parish': 'Jefferson',
                                  'St Charles Parish': 'St Charles',
                                  'St Bernard Parish': 'St Bernard',
                                  'Plaquemines Parish': 'Plaquemines',
                                  'St John the Baptist Parish': 'St John the Baptist',
                                  'St James Parish': 'St James',
                                  'Orleans Parish': 'Orleans',
                                  'Lafourche Parish': 'Lafourche',
                                   'Assumption Parish': 'Assumption',
                                  'St Mary Parish': 'St Mary',
                                  'Terrebonne Parish': 'Terrebonne',
                                   'Ascension Parish': 'Ascension',
                                  'Tangipahoa Parish': 'Tangipahoa',
                                   'St Tammany Parish': 'St Tammany',
                                  'Washington Parish': 'Washington',
                                   'St Helena Parish': 'St Helena',
                                  'Livingston Parish': 'Livingston',
                                  'Lafayette Parish': 'Lafayette',
                                  'Vermilion Parish': 'Vermilion',
                                  'St Landry Parish': 'St Landry',
                                  'Iberia Parish': 'Iberia',
                                  'Evangebar Parish': 'Evangebar',
                                  'Acadia Parish': 'Acadia',
                                  'St Martin Parish': 'St Martin',
                                   'Jefferson Davis Parish': 'Jefferson Davis',
                                  'St. Landry Parish': 'St. Landry',
                                  'Calcasieu Parish': 'Calcasieu',
                                  'Cameron Parish': 'Cameron',
                                  'Beauregard Parish': 'Beauregard',
                                  'Allen Parish': 'Allen',
                                  'Vernon Parish': 'Vernon',
                                  'East Baton Rouge Parish':'East Baton Rouge' ,
                                  'West Baton Rouge Parish': 'West Baton Rouge',
                                  'West Feliciana Parish': 'West Feliciana',
                                   'Pointe Coupee Parish': 'Pointe Coupee',
                                  'Iberville Parish': 'Iberville',
                                  'East Feliciana Parish': 'East Feliciana',
                                  'Bienville Parish': 'Bienville',
                                  'Natchitoches Parish': 'Natchitoches',
                                  'Claiborne Parish': 'Claiborne',
                                  'Caddo Parish': 'Caddo',
                                  'Bossier Parish': 'Bossier',
                                  'Webster Parish': 'Webster',
                                  'Red River Parish': 'Red River',
                                  'De Soto Parish': 'De Soto',
                                  'Sabine Parish': 'Sabine',
                                  'Ouachita Parish': 'Ouachita',
                                  'Richland Parish': 'Richland',
                                  'Franklin Parish': 'Franklin',
                                  'Morehouse Parish': 'Morehouse',
                                  'Union Parish': 'Union',
                                  'Jackson Parish': 'Jackson',
                                  'Lincoln Parish': 'Lincoln',
                                  'Madison Parish': 'Madison',
                                  'West Carroll Parish': 'West Carroll',
                                  'East Carroll Parish': 'East Carroll',
                                  'Rapides Parish': 'Rapides',
                                  'Concordia Parish': 'Concordia',
                                  'Avoyelles Parish': 'Avoyelles',
                                  'Tensas Parish': 'Tensas',
                                  'Catahoula Parish': 'Catahoula',
                                  'La Salle Parish': 'La Salle',
                                  'Winn Parish': 'Winn',
                                  'Grant Parish': 'Grant',
                                  'Caldwell Parish': 'Caldwell'}, inplace=True)

    county_latlong['county'] = county_latlong['county'].str.lower()

    df = pd.merge(df,county_latlong,on=['county','state_abbr'])
    df.drop(columns=['region','state_abbr','zip','primary_city'], inplace=True)
    df = pd.merge(df,regions,on='state')
    df = df.groupby(['date', 'region','state_abbr', 'state', 'fips','county']).median().reset_index()

    metro = pd.read_csv( metro_file,dtype={'GEOID':'str'})
    metro = metro[metro['MSA_TYPE'] == 'Metropolitan Statistical Area']

    metro_latlong = metro.groupby('CBSA_TITLE')['INTPTLAT','INTPTLON'].median().reset_index()
    metro_latlong.rename(columns={'CBSA_TITLE':'metro_area','INTPTLAT':'metro_area_lat','INTPTLON':'metro_area_lon'}, inplace=True)

    metro = metro[['GEOID', 'CBSA_TITLE']]
    metro.rename(columns={'GEOID':'fips','CBSA_TITLE':'metro_area'},inplace=True)

    df = pd.merge(df,metro,on='fips', how='left')
    df['metro_area'] = df['metro_area'].fillna('non_metro_area')

    census_df = pd.DataFrame()
    census_df['fips'] = df['fips'].unique()

    for file in list_of_files:
        f_df = pd.read_excel( file + '.xls', dtype={'STCOU':str})
        f_df.rename(columns=col_name_dict, inplace=True)
        df_cols = [c for c in f_df.columns if c.lower()[:3] != file.lower()[:3]]
        f_df=f_df[df_cols]
        f_df.rename(columns={'STCOU':'fips'},inplace=True)
        census_df = pd.merge(census_df,f_df, on='fips')

    census_df = census_df[['fips', 'All persons under 65 years without health insurance, percent 2007',
                          'Resident population: White alone, percent  (April 1 - complete count) 2010',
                          'Land area in square miles 2010','Resident population (April 1 - complete count) 2010']]
    census_df.rename(columns={'All persons under 65 years without health insurance, percent 2007':'pct_uninsured',
                             'Resident population: White alone, percent  (April 1 - complete count) 2010':'pct_white',
                              'Land area in square miles 2010':'land_sqmi',
                              'Resident population (April 1 - complete count) 2010':'population'
                             },inplace=True)

    df = pd.merge(df,census_df)
    df = pd.merge(df, regions_latlong,on='region',how='left')
    df = pd.merge(df, metro_latlong,on='metro_area',how='left')

    df['pct_nonwhite'] = 100 - df['pct_white']
    df['pct_insured'] = 100 - df['pct_uninsured']
    df['pop_per_sqmi'] = df['population']/df['land_sqmi']

    df['county_st'] = df['county'] + ", " + df['state_abbr']

    return  df

# function to make region graphs
def region_grapher(df):
    global region_input

    # create a column sof weights for population based metrics
    metrics_df = df.drop(columns=['date', 'cases', 'deaths'])
    metrics_df = metrics_df.drop_duplicates()
    metrics_df['pct_white_wt_region'] = metrics_df['pct_white']*metrics_df['population']
    metrics_df['pct_uninsured_wt_region'] = metrics_df['pct_uninsured']*metrics_df['population']

    region_temp1 = metrics_df.groupby('region')['land_sqmi', 'population'].sum().reset_index()
    region_temp1.rename(columns={'land_sqmi':'land_sqmi_region','population':'population_region'},inplace=True)
    region_temp1['pop_per_sqmi_region'] = round(region_temp1['population_region']/region_temp1['land_sqmi_region'])

    region_temp2 = metrics_df.groupby('region')['pct_white_wt_region', 'pct_uninsured_wt_region', 'population'].sum().reset_index()
    region_temp2['pct_white_region'] = round(region_temp2['pct_white_wt_region']/region_temp2['population'])
    region_temp2['pct_uninsured_region'] = round(region_temp2['pct_uninsured_wt_region']/region_temp2['population'])
    region_temp2.drop(columns=['pct_white_wt_region','pct_uninsured_wt_region','population'], inplace=True)

    region_temp2['pct_nonwhite_region'] = 100 - region_temp2['pct_white_region']
    region_temp2['pct_insured_region'] = 100 - region_temp2['pct_uninsured_region']

    region_df = df.groupby(['date','region']).agg({'cases':'sum', 'deaths':'sum', 'region_lat':'mean','region_lon':'mean'}).reset_index()

    region_data = pd.merge(region_df,region_temp1, on='region', how='left')
    region_data = pd.merge(region_data,region_temp2, on='region', how='left')
    region_data['cases_per_mil'] = round(region_data['cases']/region_data['population_region']*1000000)
    region_data['deaths_per_mil'] = round(region_data['deaths']/region_data['population_region']*1000000)

    if not region_input:
        # filter to two previous dates and create column indicating region hasn't changed
        map1_df = region_data[region_data['date'].isin(last_two_dates)]
        map1_df = map1_df.sort_values(by=['region','date'])
        map1_df['same_as_prev'] = map1_df['region'].shift(1) == map1_df['region']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100


    else:
        # filter to two previous dates and regions selected then create column indicating region hasn't changed
        map1_df = region_data[region_data['date'].isin(last_two_dates) & region_data['region'].isin(region_input)]
        map1_df = map1_df.sort_values(by=['region','date'])
        map1_df['same_as_prev'] = map1_df['region'].shift(1) == map1_df['region']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100


    # take only rows where region is same as previous row
    map1_df = map1_df[map1_df['same_as_prev'] == True]

    if not region_input:
        # map the region values back onto geo_df with each county
        region_map_df = pd.merge(geo_df, map1_df, on='region', how='left')
        lat = region_map_df['region_lat'].mean()
        lon = region_map_df['region_lon'].mean()-5

    else:
        # map the region values back onto geo_df with each county
        region_map_df = pd.merge(geo_df, map1_df, on='region', how='left')
        region_map_df = region_map_df[region_map_df['region'].isin(region_input)]
        lat = region_map_df['region_lat'].mean()
        lon = region_map_df['region_lon'].mean()

    if len(map1_df['region'].unique()) == 1:
        zoom = 4.0
    elif len(map1_df['region'].unique()) == 2:
        zoom = 3.5
    elif len(map1_df['region'].unique()) == 3:
        zoom = 3.2
    elif len(map1_df['region'].unique()) == 4:
        zoom = 2.9
    else:
        zoom = 2.6

    map1 = px.choropleth_mapbox(region_map_df, geojson=counties, locations='fips', color=color,
                                color_continuous_scale="reds", mapbox_style="carto-positron",
                                zoom=zoom, center = {"lat": lat, "lon": lon},
                                opacity=0.8, hover_name='region')
    #map1.update_traces(marker_bar_width=0)

    # bar graph
    if not region_input:
        # filter to two previous dates and create column indicating region hasn't changed
        bar1_df = region_data[region_data['date'].isin(dates_to_use)]
        bar1_df = bar1_df.sort_values(by=['region','date'])
        bar1_df['same_as_prev'] = bar1_df['region'].shift(1) == bar1_df['region']

    else:
        # filter to two previous dates and regions selected then create column indicating region hasn't changed
        bar1_df = region_data[region_data['date'].isin(dates_to_use) & region_data['region'].isin(region_input)]
        bar1_df = bar1_df.sort_values(by=['region','date'])
        bar1_df['same_as_prev'] = bar1_df['region'].shift(1) == bar1_df['region']

    bar1_df = bar1_df[bar1_df['same_as_prev'] == True]

    bar1_df = bar1_df.groupby('date').agg({'cases':'sum','deaths':'sum','population_region':'sum'}).reset_index()
    bar1_df['cases_per_mil'] = round(bar1_df['cases']/bar1_df['population_region']*1000000)
    bar1_df['deaths_per_mil'] = round(bar1_df['deaths']/bar1_df['population_region']*1000000)
    bar1_df['change'] = bar1_df[metric_input].diff()
    bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100

    bar1 = px.bar(bar1_df, x='date', y=color) # end make bar graph

    # make the heatmap
    hm1_df = region_data.sort_values(by=['region','date'])
    hm1_df = hm1_df[hm1_df['date'].isin(dates_to_use)]
    hm1_df['change'] = hm1_df[metric_input].diff()
    hm1_df['percent_chg'] = (hm1_df[metric_input] - hm1_df[metric_input].shift(1))/hm1_df[metric_input].shift(1)*100
    hm1_df['same_as_prev'] = hm1_df['region'].shift(1) == hm1_df['region']
    hm1_df = hm1_df[hm1_df['same_as_prev'] == True]

    if not region_input:
        # get largest
        region_input = region_list
        hm1_df = hm1_df[hm1_df['region'].isin(region_input)]
        addl_region_input = []

    elif len(region_input) < 2:
        # begin the closest code
        comp_df = region_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_region', 'pct_insured_region', 'land_sqmi_region'])
        comp_df = comp_df.groupby('region').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_region', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_region', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_region', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='region_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='region_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)

        temp = comp_df[comp_df['region'].isin(region_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['region']
        b = closest

        d = {'region':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance',inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_region_input = dist_list[:no_scatter]

        hm1_df = hm1_df[(hm1_df['region'].isin(addl_region_input)) | (hm1_df['region'].isin(region_input))]

    else:
        # get only those selected
        hm1_df = hm1_df[hm1_df['region'].isin(region_input)]
        addl_region_input = []

    hm_dates = hm1_df['date']
    hm_geos = hm1_df['region']
    hm_values = hm1_df[color]
    ticks = len(hm1_df['date'].unique())

    hm1 = go.Figure(data=go.Heatmap(
            z=hm_values,
            x=hm_dates,
            y=hm_geos,
            colorscale='reds'))

    hm1.update_layout(
        title=color,
        xaxis_nticks=ticks)

    # end make heatmap

    if not region_input:
        region_input = region_list
        scat1_df = region_data[region_data['region'].isin(region_input)]
        scat1_df = scat1_df[scat1_df['date'] == region_data['date'].max()]

    elif len(region_input) < 2:
# begin the closest code
        comp_df = region_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_region', 'pct_insured_region', 'land_sqmi_region'])
        comp_df = comp_df.groupby('region').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_region', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_region', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_region', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='region_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='region_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)

        temp = comp_df[comp_df['region'].isin(region_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['region']
        b = closest

        d = {'region':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance',inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_region_input = dist_list[:no_scatter]
# end closest code
        scat1_df = region_data[(region_data['region'].isin(addl_region_input)) | (region_data['region'].isin(region_input))]
        scat1_df = scat1_df[scat1_df['date'] == region_data['date'].max()]

    else:
        scat1_df = region_data[region_data['region'].isin(region_input)]
        scat1_df = scat1_df[scat1_df['date'] == region_data['date'].max()]

    scat1 = px.scatter(scat1_df,x=scatter_x, y=scatter_y, size = scatter_sz, color = scatter_col, hover_name='region', trendline='ols', color_continuous_scale='YlOrRd')

    return map1, bar1, hm1, scat1

# function to make state graphs
def state_grapher(df):
    global state_input

# create a column sof weights for population based metrics
    metrics_df = df.drop(columns=['date', 'cases', 'deaths'])
    metrics_df = metrics_df.drop_duplicates()
    metrics_df['pct_white_wt_state'] = metrics_df['pct_white']*metrics_df['population']
    metrics_df['pct_uninsured_wt_state'] = metrics_df['pct_uninsured']*metrics_df['population']

    state_temp1 = metrics_df.groupby('state')['land_sqmi', 'population'].sum().reset_index()

    state_by_pop = state_temp1.sort_values(by='population', ascending=False)
    state_by_pop = state_by_pop.iloc[:no_scatter,0].tolist()

    state_temp1.rename(columns={'land_sqmi':'land_sqmi_state','population':'population_state'},inplace=True)
    state_temp1['pop_per_sqmi_state'] = round(state_temp1['population_state']/state_temp1['land_sqmi_state'])

    state_temp2 = metrics_df.groupby('state')['pct_white_wt_state', 'pct_uninsured_wt_state', 'population'].sum().reset_index()
    state_temp2['pct_white_state'] = round(state_temp2['pct_white_wt_state']/state_temp2['population'])
    state_temp2['pct_uninsured_state'] = round(state_temp2['pct_uninsured_wt_state']/state_temp2['population'])
    state_temp2.drop(columns=['pct_white_wt_state','pct_uninsured_wt_state','population'], inplace=True)

    state_temp2['pct_nonwhite_state'] = 100 - state_temp2['pct_white_state']
    state_temp2['pct_insured_state'] = 100 - state_temp2['pct_uninsured_state']

    state_df = df.groupby(['date','state']).agg({'cases':'sum', 'deaths':'sum', 'state_lat':'mean','state_lon':'mean'}).reset_index()

    state_data = pd.merge(state_df,state_temp1, on='state', how='left')
    state_data = pd.merge(state_data,state_temp2, on='state', how='left')
    state_data['cases_per_mil'] = round(state_data['cases']/state_data['population_state']*1000000)
    state_data['deaths_per_mil'] = round(state_data['deaths']/state_data['population_state']*1000000)

    if not state_input:

        # filter to two previous dates and create column indicating region hasn't changed
        map1_df = state_data[state_data['date'].isin(last_two_dates)]
        map1_df = map1_df.sort_values(by=['state','date'])
        map1_df['same_as_prev'] = map1_df['state'].shift(1) == map1_df['state']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100


    else:

        # filter to two previous dates and regions selected then create column indicating region hasn't changed
        map1_df = state_data[state_data['date'].isin(last_two_dates) & state_data['state'].isin(state_input)]
        map1_df = map1_df.sort_values(by=['state','date'])
        map1_df['same_as_prev'] = map1_df['state'].shift(1) == map1_df['state']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100

    # take only rows where region is same as previous row
    map1_df = map1_df[map1_df['same_as_prev'] == True]

    if not state_input:

        # map the region values back onto geo_df with each county
        state_map_df = pd.merge(geo_df, map1_df, on='state', how='left')
        lat = state_map_df['state_lat'].mean()
        lon = state_map_df['state_lon'].mean()-5

    else:

        # map the region values back onto geo_df with each county
        state_map_df = pd.merge(geo_df, map1_df, on='state', how='left')
        state_map_df = state_map_df[state_map_df['state'].isin(state_input)]
        lat = state_map_df['state_lat'].mean()
        lon = state_map_df['state_lon'].mean()

    if len(map1_df['state'].unique()) == 1:
        zoom = 5
    elif len(map1_df['state'].unique()) == 2:
        zoom = 4.5
    elif len(map1_df['state'].unique()) == 3:
        zoom = 3.8
    elif len(map1_df['state'].unique()) == 4:
        zoom = 3.2
    else:
        zoom = 2.6

    map1 = px.choropleth_mapbox(state_map_df, geojson=counties, locations='fips', color=color,
                                       color_continuous_scale="reds",
                                       mapbox_style="carto-positron",
                                       zoom=zoom, center = {"lat": lat, "lon": lon},
                                       opacity=0.8, hover_name='state')
    #map1.update_traces(marker_bar_width=0)

    if not state_input:
        # filter to two previous dates and create column indicating state hasn't changed
        bar1_df = state_data[state_data['date'].isin(dates_to_use)]
        bar1_df = bar1_df.sort_values(by=['state','date'])
        bar1_df['same_as_prev'] = bar1_df['state'].shift(1) == bar1_df['state']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100


    else:
        # filter to two previous dates and states selected then create column indicating state hasn't changed
        bar1_df = state_data[state_data['date'].isin(dates_to_use) & state_data['state'].isin(state_input)]
        bar1_df = bar1_df.sort_values(by=['state','date'])
        bar1_df['same_as_prev'] = bar1_df['state'].shift(1) == bar1_df['state']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = round((bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100)

    bar1_df = bar1_df[bar1_df['same_as_prev'] == True]

    bar1_df = bar1_df.groupby('date').agg({'cases':'sum','deaths':'sum','population_state':'sum'}).reset_index()
    bar1_df['cases_per_mil'] = round(bar1_df['cases']/bar1_df['population_state']*1000000)
    bar1_df['deaths_per_mil'] = round(bar1_df['deaths']/bar1_df['population_state']*1000000)
    bar1_df['change'] = bar1_df[metric_input].diff()
    bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100

    bar1 = px.bar(bar1_df, x='date', y=color) # end of make bar graph

    # make the heatmap
    hm1_df = state_data.sort_values(by=['state','date'])
    hm1_df = hm1_df[hm1_df['date'].isin(dates_to_use)]
    hm1_df['change'] = hm1_df[metric_input].diff()
    hm1_df['percent_chg'] = (hm1_df[metric_input] - hm1_df[metric_input].shift(1))/hm1_df[metric_input].shift(1)*100
    hm1_df['same_as_prev'] = hm1_df['state'].shift(1) == hm1_df['state']
    hm1_df = hm1_df[hm1_df['same_as_prev'] == True]

    if not state_input:
        # get largest
        hm1_df = hm1_df[hm1_df['state'].isin(state_by_pop)]
        addl_state_input = []

    elif len(state_input) < 2:
        # begin the closest code
        comp_df = state_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_state', 'pct_insured_state', 'land_sqmi_state'])
        comp_df = comp_df.groupby('state').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_state', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_state', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_state', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='state_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='state_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)

        temp = comp_df[comp_df['state'].isin(state_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['state']
        b = closest

        d = {'state':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance',inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_state_input = dist_list[:no_scatter]

        hm1_df = hm1_df[(hm1_df['state'].isin(addl_state_input)) | (hm1_df['state'].isin(state_input))]

    else:
        # get only those selected
        hm1_df = hm1_df[hm1_df['state'].isin(state_input)]
        addl_state_input = []

    hm_dates = hm1_df['date']
    hm_geos = hm1_df['state']
    hm_values = hm1_df[color]
    ticks = len(hm1_df['date'].unique())

    hm1 = go.Figure(data=go.Heatmap(
            z=hm_values,
            x=hm_dates,
            y=hm_geos,
            colorscale='reds'))

    hm1.update_layout(
        title=color,
        xaxis_nticks=ticks)

    # end make heatmap



    if not state_input:
        scat1_df = state_data[state_data['state'].isin(state_by_pop)]
        scat1_df = scat1_df[scat1_df['date'] == state_data['date'].max()]

    elif len(state_input) < 2:
        # begin the closest code
        comp_df = state_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_state', 'pct_insured_state', 'land_sqmi_state'])
        comp_df = comp_df.groupby('state').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_state', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_state', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_state', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='state_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='state_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)
        comp_df.sort_values(by='state', inplace=True)

        temp = comp_df[comp_df['state'].isin(state_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['state']
        b = closest

        d = {'state':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance', inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_state_input = dist_list[:no_scatter]
        # end closest code
        scat1_df = state_data[state_data['state'].isin(state_input) | state_data['state'].isin(addl_state_input)]
        scat1_df = scat1_df[scat1_df['date'] == scat1_df['date'].max()]

    else:
        scat1_df = state_data[state_data['state'].isin(state_input)]
        scat1_df = scat1_df[scat1_df['date'] == scat1_df['date'].max()]

    scat1 = px.scatter(scat1_df,x=scatter_x, y=scatter_y, size = scatter_sz, color = scatter_col,
                       hover_name='state', trendline='ols', color_continuous_scale='YlOrRd')

    return map1, bar1, hm1, scat1

# function to make metro_area graphs

def metro_grapher(df):
    global metro_input

    # create a column sof weights for population based metrics
    metrics_df = df.drop(columns=['date', 'cases', 'deaths'])
    metrics_df = metrics_df.drop_duplicates()
    metrics_df['pct_white_wt_metro_area'] = metrics_df['pct_white']*metrics_df['population']
    metrics_df['pct_uninsured_wt_metro_area'] = metrics_df['pct_uninsured']*metrics_df['population']

    metro_area_temp1 = metrics_df.groupby('metro_area')['land_sqmi', 'population'].sum().reset_index()

    metro_by_pop = metro_area_temp1.sort_values(by='population', ascending=False)
    metro_by_pop = metro_by_pop[metro_by_pop['metro_area'] != 'non_metro_area']
    metro_by_pop = metro_by_pop.iloc[:no_scatter,0].tolist()

    metro_area_temp1.rename(columns={'land_sqmi':'land_sqmi_metro_area','population':'population_metro_area'},inplace=True)
    metro_area_temp1['pop_per_sqmi_metro_area'] = round(metro_area_temp1['population_metro_area']/metro_area_temp1['land_sqmi_metro_area'])

    metro_area_temp2 = metrics_df.groupby('metro_area')['pct_white_wt_metro_area', 'pct_uninsured_wt_metro_area', 'population'].sum().reset_index()
    metro_area_temp2['pct_white_metro_area'] = round(metro_area_temp2['pct_white_wt_metro_area']/metro_area_temp2['population'])
    metro_area_temp2['pct_uninsured_metro_area'] = round(metro_area_temp2['pct_uninsured_wt_metro_area']/metro_area_temp2['population'])
    metro_area_temp2.drop(columns=['pct_white_wt_metro_area','pct_uninsured_wt_metro_area','population'], inplace=True)

    metro_area_temp2['pct_nonwhite_metro_area'] = 100 - metro_area_temp2['pct_white_metro_area']
    metro_area_temp2['pct_insured_metro_area'] = 100 - metro_area_temp2['pct_uninsured_metro_area']

    metro_area_df = df.groupby(['date','metro_area']).agg({'cases':'sum', 'deaths':'sum', 'metro_area_lat':'mean','metro_area_lon':'mean'}).reset_index()

    metro_area_data = pd.merge(metro_area_df,metro_area_temp1, on='metro_area', how='left')
    metro_area_data = pd.merge(metro_area_data,metro_area_temp2, on='metro_area', how='left')
    metro_area_data['cases_per_mil'] = round(metro_area_data['cases']/metro_area_data['population_metro_area']*1000000)
    metro_area_data['deaths_per_mil'] = round(metro_area_data['deaths']/metro_area_data['population_metro_area']*1000000)

    if not metro_input:
        # filter to two previous dates and create column indicating region hasn't changed
        map1_df = metro_area_data[metro_area_data['date'].isin(last_two_dates)]
        map1_df = map1_df.sort_values(by=['metro_area','date'])
        map1_df['same_as_prev'] = map1_df['metro_area'].shift(1) == map1_df['metro_area']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100

    else:
        # filter to two previous dates and regions selected then create column indicating region hasn't changed
        map1_df = metro_area_data[metro_area_data['date'].isin(last_two_dates) & metro_area_data['metro_area'].isin(metro_input)]
        map1_df = map1_df.sort_values(by=['metro_area','date'])
        map1_df['same_as_prev'] = map1_df['metro_area'].shift(1) == map1_df['metro_area']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100

    # take only rows where region is same as previous row
    map1_df = map1_df[map1_df['same_as_prev'] == True]

    if not metro_input:
        # map the region values back onto geo_df with each county
        metro_area_map_df = pd.merge(geo_df, map1_df, on='metro_area', how='left')
        metro_area_map_df = metro_area_map_df[metro_area_map_df['metro_area'] != 'non_metro_area']
        lat = metro_area_map_df['metro_area_lat'].mean()
        lon = metro_area_map_df['metro_area_lon'].mean()-5
    else:

        # map the region values back onto geo_df with each county
        metro_area_map_df = pd.merge(geo_df, map1_df, on='metro_area', how='left')
        metro_area_map_df = metro_area_map_df[metro_area_map_df['metro_area'].isin(metro_input)]
        lat = metro_area_map_df['metro_area_lat'].mean()
        lon = metro_area_map_df['metro_area_lon'].mean()

    if len(map1_df['metro_area'].unique()) == 1:
        zoom = 6.0
    elif len(map1_df['metro_area'].unique()) == 2:
        zoom = 5.5
    elif len(map1_df['metro_area'].unique()) == 3:
        zoom = 4.8
    elif len(map1_df['metro_area'].unique()) == 4:
        zoom = 4.2
    else:
        zoom = 2.6

    map1 = px.choropleth_mapbox(metro_area_map_df, geojson=counties, locations='fips', color=color,
                                       color_continuous_scale="reds",
                                       mapbox_style="carto-positron",
                                       zoom=zoom, center = {"lat": lat, "lon": lon},
                                       opacity=0.8, hover_name='metro_area')

    #map1.update_traces(marker_bar_width=0)

    if not metro_input:

        # filter to two previous dates and create column indicating metro_area hasn't changed
        bar1_df = metro_area_data[metro_area_data['date'].isin(dates_to_use)]
        bar1_df = bar1_df[bar1_df['metro_area'] != 'non_metro_area']
        bar1_df = bar1_df.sort_values(by=['metro_area','date'])
        bar1_df['same_as_prev'] = bar1_df['metro_area'].shift(1) == bar1_df['metro_area']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100


    else:

        # filter to two previous dates and metro_areas selected then create column indicating metro_area hasn't changed
        bar1_df = metro_area_data[metro_area_data['date'].isin(dates_to_use) & metro_area_data['metro_area'].isin(metro_input)]
        bar1_df = bar1_df.sort_values(by=['metro_area','date'])
        bar1_df['same_as_prev'] = bar1_df['metro_area'].shift(1) == bar1_df['metro_area']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = round((bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100)

    bar1_df = bar1_df[bar1_df['same_as_prev'] == True]

    bar1_df = bar1_df.groupby('date').agg({'cases':'sum','deaths':'sum','population_metro_area':'sum'}).reset_index()
    bar1_df['cases_per_mil'] = round(bar1_df['cases']/bar1_df['population_metro_area']*1000000)
    bar1_df['deaths_per_mil'] = round(bar1_df['deaths']/bar1_df['population_metro_area']*1000000)
    bar1_df['change'] = bar1_df[metric_input].diff()
    bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100

    bar1 = px.bar(bar1_df, x='date', y=color) # end of make bar graph

    # make the heatmap
    hm1_df = metro_area_data.sort_values(by=['metro_area','date'])
    hm1_df = hm1_df[hm1_df['date'].isin(dates_to_use)]
    hm1_df['change'] = hm1_df[metric_input].diff()
    hm1_df['percent_chg'] = (hm1_df[metric_input] - hm1_df[metric_input].shift(1))/hm1_df[metric_input].shift(1)*100
    hm1_df['same_as_prev'] = hm1_df['metro_area'].shift(1) == hm1_df['metro_area']
    hm1_df = hm1_df[hm1_df['same_as_prev'] == True]

    if not metro_input:
        # get largest
        hm1_df = hm1_df[hm1_df['metro_area'].isin(metro_by_pop)]
        addl_metro_input = []

    elif len(metro_input) < 2:
        # begin the closest code
        comp_df = metro_area_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_metro_area', 'pct_insured_metro_area', 'land_sqmi_metro_area'])
        comp_df = comp_df.groupby('metro_area').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_metro_area', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_metro_area', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_metro_area', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='metro_area_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='metro_area_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)

        temp = comp_df[comp_df['metro_area'].isin(metro_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['metro_area']
        b = closest

        d = {'metro_area':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance',inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_state_input = dist_list[:no_scatter]

        hm1_df = hm1_df[(hm1_df['metro_area'].isin(addl_metro_input)) | (hm1_df['metro_area'].isin(metro_input))]

    else:
        # get only those selected
        hm1_df = hm1_df[hm1_df['metro_area'].isin(metro_input)]
        addl_metro_input = []

    hm_dates = hm1_df['date']
    hm_geos = hm1_df['metro_area']
    hm_values = hm1_df[color]
    ticks = len(hm1_df['date'].unique())

    hm1 = go.Figure(data=go.Heatmap(
            z=hm_values,
            x=hm_dates,
            y=hm_geos,
            colorscale='reds'))

    hm1.update_layout(
        title=color,
        xaxis_nticks=ticks)

    # end make heatmap



    if not metro_input:

        scat1_df = metro_area_data[metro_area_data['date'] == metro_area_data['date'].max()]
        scat1_df = scat1_df[scat1_df['metro_area'] != 'non_metro_area']
        scat1_df = scat1_df[scat1_df['metro_area'].isin(metro_by_pop)]

    else:

        scat1_df = metro_area_data[metro_area_data['date'] == metro_area_data['date'].max()]

        if len(metro_input) < 2:
            # begin the closest code
            comp_df = metro_area_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_metro_area', 'pct_insured_metro_area', 'land_sqmi_metro_area'])
            comp_df = comp_df.groupby('metro_area').max().reset_index()

            comp_df.sort_values(by='pct_uninsured_metro_area', inplace=True)
            comp_df['uninsured_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='pct_nonwhite_metro_area', inplace=True)
            comp_df['nonwhite_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='pop_per_sqmi_metro_area', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='metro_area_lat', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='metro_area_lon', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.reset_index(inplace=True, drop=True)
            comp_df.sort_values(by='metro_area', inplace=True)

            temp = comp_df[comp_df['metro_area'].isin(metro_input)]

            closest = []

            lat1 = temp.iloc[0,1] # need to ref latitude of region_input
            long1 = temp.iloc[0,2] # need to ref longitude of region_input

            for i in range(len(comp_df)):
                lat2 = comp_df.iloc[i,1]
                long2 = comp_df.iloc[i,2]

                d = dist(lat1, long1, lat2, long2)

                closest.append(d)

            a = comp_df['metro_area']
            b = closest

            d = {'metro_area':a,'distance':b}

            distance = pd.DataFrame(d)
            distance.sort_values(by='distance', inplace=True)

            dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
            dist_col = dist_col[1:101]

            dist_list=[]
            for k in range(len(dist_col)):
                a = dist_col[k][0]
                dist_list.append(a)

            addl_metro_input = dist_list[:no_scatter]
            # end closest code

            scat1_df = scat1_df[scat1_df['metro_area'].isin(metro_input) | scat1_df['metro_area'].isin(addl_metro_input)]
        else:
            scat1_df = scat1_df[scat1_df['metro_area'].isin(metro_input)]

    scat1 = px.scatter(scat1_df,x=scatter_x, y=scatter_y, size = scatter_sz, color = scatter_col,
                       hover_name='metro_area', trendline='ols', color_continuous_scale='YlOrRd')

    return map1, bar1, hm1, scat1

# function to make county graphs

def county_grapher(df):
    global county_input

    # create a column sof weights for population based metrics
    metrics_df = df.drop(columns=['date', 'cases', 'deaths'])
    metrics_df = metrics_df.drop_duplicates()
    metrics_df['pct_white_wt_county'] = metrics_df['pct_white']*metrics_df['population']
    metrics_df['pct_uninsured_wt_county'] = metrics_df['pct_uninsured']*metrics_df['population']

    county_temp1 = metrics_df.groupby('county_st')['land_sqmi', 'population'].sum().reset_index()

    county_by_pop = county_temp1.sort_values(by='population', ascending=False)
    county_by_pop = county_by_pop[county_by_pop['county_st'] != 'non_metro_area']
    county_by_pop = county_by_pop.iloc[:no_scatter,0].tolist()

    county_temp1.rename(columns={'land_sqmi':'land_sqmi_county','population':'population_county'},inplace=True)
    county_temp1['pop_per_sqmi_county'] = round(county_temp1['population_county']/county_temp1['land_sqmi_county'])

    county_temp2 = metrics_df.groupby('county_st')['pct_white_wt_county', 'pct_uninsured_wt_county', 'population'].sum().reset_index()
    county_temp2['pct_white_county'] = round(county_temp2['pct_white_wt_county']/county_temp2['population'])
    county_temp2['pct_uninsured_county'] = round(county_temp2['pct_uninsured_wt_county']/county_temp2['population'])
    county_temp2.drop(columns=['pct_white_wt_county','pct_uninsured_wt_county','population'], inplace=True)

    county_temp2['pct_nonwhite_county'] = 100 - county_temp2['pct_white_county']
    county_temp2['pct_insured_county'] = 100 - county_temp2['pct_uninsured_county']

    county_df = df.groupby(['date','county_st']).agg({'cases':'sum', 'deaths':'sum', 'county_lat':'mean','county_lon':'mean'}).reset_index()

    county_data = pd.merge(county_df,county_temp1, on='county_st', how='left')
    county_data = pd.merge(county_data,county_temp2, on='county_st', how='left')
    county_data['cases_per_mil'] = round(county_data['cases']/county_data['population_county']*1000000)
    county_data['deaths_per_mil'] = round(county_data['deaths']/county_data['population_county']*1000000)

    if not county_input:

        # filter to two previous dates and create column indicating region hasn't changed
        map1_df = county_data[county_data['date'].isin(last_two_dates)]
        map1_df = map1_df.sort_values(by=['county_st','date'])
        map1_df['same_as_prev'] = map1_df['county_st'].shift(1) == map1_df['county_st']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100

    else:

        # filter to two previous dates and regions selected then create column indicating region hasn't changed
        map1_df = county_data[county_data['date'].isin(last_two_dates) & county_data['county_st'].isin(county_input)]
        map1_df = map1_df.sort_values(by=['county_st','date'])
        map1_df['same_as_prev'] = map1_df['county_st'].shift(1) == map1_df['county_st']

        # make new columns for graphs to note changes between two dates
        map1_df['change'] = map1_df[metric_input].diff()
        map1_df['change_pct'] = (map1_df[metric_input] - map1_df[metric_input].shift(1))/map1_df[metric_input].shift(1)*100

    # take only rows where region is same as previous row
    map1_df = map1_df[map1_df['same_as_prev'] == True]

    if not county_input:

        # map the region values back onto geo_df with each county
        county_map_df = pd.merge(geo_df, map1_df, on='county_st', how='left')
        lat = county_map_df['county_lat'].mean()
        lon = county_map_df['county_lon'].mean()-5

    else:

        # map the region values back onto geo_df with each county
        county_map_df = pd.merge(geo_df, map1_df, on='county_st', how='left')
        county_map_df = county_map_df[county_map_df['county_st'].isin(county_input)]
        lat = county_map_df['county_lat'].mean()
        lon = county_map_df['county_lon'].mean()

    if len(map1_df['county_st'].unique()) == 1:
        zoom = 6.5
    elif len(map1_df['county_st'].unique()) == 2:
        zoom = 5.5
    elif len(map1_df['county_st'].unique()) == 3:
        zoom = 4.8
    elif len(map1_df['county_st'].unique()) == 4:
        zoom = 4.2
    else:
        zoom = 2.6

    map1 = px.choropleth_mapbox(county_map_df, geojson=counties, locations='fips', color=color,
                                       color_continuous_scale="reds",
                                       mapbox_style="carto-positron",
                                       zoom=zoom, center = {"lat": lat, "lon": lon},
                                       opacity=0.8, hover_name='county_st')

    if not county_input:

        # filter to two previous dates and create column indicating county hasn't changed
        bar1_df = county_data[county_data['date'].isin(dates_to_use)]
        bar1_df = bar1_df.sort_values(by=['county_st','date'])
        bar1_df['same_as_prev'] = bar1_df['county_st'].shift(1) == bar1_df['county_st']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100


    else:

        # filter to two previous dates and countys selected then create column indicating county hasn't changed
        bar1_df = county_data[county_data['date'].isin(dates_to_use) & county_data['county_st'].isin(county_input)]
        bar1_df = bar1_df.sort_values(by=['county_st','date'])
        bar1_df['same_as_prev'] = bar1_df['county_st'].shift(1) == bar1_df['county_st']

        # make new columns for graphs to note changes between two dates
        bar1_df['change'] = bar1_df[metric_input].diff()
        bar1_df['change_pct'] = round((bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100)

    bar1_df = bar1_df[bar1_df['same_as_prev'] == True]

    bar1_df = bar1_df.groupby('date').agg({'cases':'sum','deaths':'sum','population_county':'sum'}).reset_index()
    bar1_df['cases_per_mil'] = round(bar1_df['cases']/bar1_df['population_county']*1000000)
    bar1_df['deaths_per_mil'] = round(bar1_df['deaths']/bar1_df['population_county']*1000000)
    bar1_df['change'] = bar1_df[metric_input].diff()
    bar1_df['change_pct'] = (bar1_df[metric_input] - bar1_df[metric_input].shift(1))/bar1_df[metric_input].shift(1)*100

    bar1 = px.bar(bar1_df, x='date', y=color) # end make bar graph

    # make the heatmap
    hm1_df = county_data.sort_values(by=['county_st','date'])
    hm1_df = hm1_df[hm1_df['date'].isin(dates_to_use)]
    hm1_df['change'] = hm1_df[metric_input].diff()
    hm1_df['percent_chg'] = (hm1_df[metric_input] - hm1_df[metric_input].shift(1))/hm1_df[metric_input].shift(1)*100
    hm1_df['same_as_prev'] = hm1_df['county_st'].shift(1) == hm1_df['county_st']
    hm1_df = hm1_df[hm1_df['same_as_prev'] == True]

    if not county_input:
        # get largest
        hm1_df = hm1_df[hm1_df['county_st'].isin(county_by_pop)]
        addl_county_input = []

    elif len(county_input) < 2:
        # begin the closest code
        comp_df = county_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_county', 'pct_insured_county', 'land_sqmi_county'])
        comp_df = comp_df.groupby('county_st').max().reset_index()

        comp_df.sort_values(by='pct_uninsured_county', inplace=True)
        comp_df['uninsured_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pct_nonwhite_county', inplace=True)
        comp_df['nonwhite_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='pop_per_sqmi_county', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='county_lat', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.sort_values(by='county_lon', inplace=True)
        comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

        comp_df.reset_index(inplace=True, drop=True)

        temp = comp_df[comp_df['county_st'].isin(county_input)]

        closest = []

        lat1 = temp.iloc[0,1] # need to ref latitude of region_input
        long1 = temp.iloc[0,2] # need to ref longitude of region_input

        for i in range(len(comp_df)):
            lat2 = comp_df.iloc[i,1]
            long2 = comp_df.iloc[i,2]

            d = dist(lat1, long1, lat2, long2)

            closest.append(d)

        a = comp_df['county_st']
        b = closest

        d = {'county_st':a,'distance':b}

        distance = pd.DataFrame(d)
        distance.sort_values(by='distance',inplace=True)

        dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
        dist_col = dist_col[1:101]

        dist_list=[]
        for k in range(len(dist_col)):
            a = dist_col[k][0]
            dist_list.append(a)

        addl_county_input = dist_list[:no_scatter]

        hm1_df = hm1_df[(hm1_df['county_st'].isin(addl_county_input)) | (hm1_df['county_st'].isin(county_input))]

    else:
        # get only those selected
        hm1_df = hm1_df[hm1_df['county_st'].isin(county_input)]
        addl_county_input = []

    hm_dates = hm1_df['date']
    hm_geos = hm1_df['county_st']
    hm_values = hm1_df[color]
    ticks = len(hm1_df['date'].unique())

    hm1 = go.Figure(data=go.Heatmap(
            z=hm_values,
            x=hm_dates,
            y=hm_geos,
            colorscale='reds'))

    hm1.update_layout(
        title=color,
        xaxis_nticks=ticks)

    # end make heatmap


    if not county_input:
        scat1_df = county_data[county_data['date'] == county_data['date'].max()]
        scat1_df = scat1_df[scat1_df['county_st'].isin(county_by_pop)]

    else:
        scat1_df = county_data[county_data['date'] == county_data['date'].max()]

        if len(county_input) < 2:
            # begin the closest code
            comp_df = county_data.drop(columns=['date', 'cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'pct_white_county', 'pct_insured_county', 'land_sqmi_county'])
            comp_df = comp_df.groupby('county_st').max().reset_index()

            comp_df.sort_values(by='pct_uninsured_county', inplace=True)
            comp_df['uninsured_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='pct_nonwhite_county', inplace=True)
            comp_df['nonwhite_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='pop_per_sqmi_county', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='county_lat', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.sort_values(by='county_lon', inplace=True)
            comp_df['pop_per_sqmi_order'] = np.arange(len(comp_df))

            comp_df.reset_index(inplace=True, drop=True)
            comp_df.sort_values(by='county_st', inplace=True)

            temp = comp_df[comp_df['county_st'].isin(county_input)]

            closest = []

            lat1 = temp.iloc[0,1] # need to ref latitude of region_input
            long1 = temp.iloc[0,2] # need to ref longitude of region_input

            for i in range(len(comp_df)):
                lat2 = comp_df.iloc[i,1]
                long2 = comp_df.iloc[i,2]

                d = dist(lat1, long1, lat2, long2)

                closest.append(d)

            a = comp_df['county_st']
            b = closest

            d = {'county_st':a,'distance':b}

            distance = pd.DataFrame(d)
            distance.sort_values(by='distance', inplace=True)

            dist_col = [tuple(r) for r in distance.to_numpy().tolist()]
            dist_col = dist_col[1:101]

            dist_list=[]
            for k in range(len(dist_col)):
                a = dist_col[k][0]
                dist_list.append(a)

            addl_county_input = dist_list[:no_scatter]
            # end closest code

            scat1_df = scat1_df[scat1_df['county_st'].isin(county_input) | scat1_df['county_st'].isin(addl_county_input)]
        else:
            scat1_df = scat1_df[scat1_df['county_st'].isin(county_input)]

    scat1 = px.scatter(scat1_df,x=scatter_x, y=scatter_y, size = scatter_sz, color = scatter_col,
                       hover_name='county_st', trendline='ols', color_continuous_scale='YlOrRd')

    return map1, bar1, hm1, scat1


df = get_data(covid_url, geo_url, geo_file,census_files)

#sidebar inputs
image = 'https://upload.wikimedia.org/wikipedia/commons/9/96/3D_medical_animation_coronavirus_structure.jpg'
st.sidebar.image(image, caption='Coronavirus Structure', use_column_width=True, format='JPEG')
st.sidebar.title('User Inputs')
st.sidebar.subheader('Choose map/bar chart elements')

# geographic data
geo_df = df[['region', 'state', 'state_abbr', 'metro_area', 'county_st', 'fips']]
geo_df = geo_df.groupby(['region','state', 'state_abbr','metro_area', 'county_st','fips']).size().reset_index()
geo_df = geo_df.iloc[:, :-1]

# dates for diffferent views
date_range = (df['date'].max() - df['date'].min()).days
days_input = st.sidebar.number_input('view change for previous _ days', value=7, max_value=1000)
date_range_days = round(date_range/days_input)

# build from days_input
dates_to_use = []

# retrieve data using other inputs
for i in range(date_range_days):
    dates_to_use.append(df['date'].max() - timedelta(days=i*days_input))
    df_pd = df[df['date'].isin(dates_to_use)]

# last dates for calculating most recent change(s)
last_two_dates = dates_to_use[:2]
last_three_dates = dates_to_use[:3]

# more user inputs

# user inputs
st.title('COVID-19 Interactive App')

metric_input = st.sidebar.radio('Choose a Metric', options=['cases','deaths'])

per_million = st.sidebar.checkbox('per million residents')
percent_chg = st.sidebar.checkbox('percent change in metric')

# graph inputs based on choices of metric and per_million
if per_million == True:

    if percent_chg == True:
        metric_input = metric_input + "_per_mil"
        color = 'change_pct'

    else:
        metric_input = metric_input + "_per_mil"
        color = 'change'

else:

    if percent_chg == True:
        color = 'change_pct'

    else:
        color='change'

# user inputs
view_input = st.sidebar.radio('Geographical Scope', options=['region','state','metro', 'county'])

region_list = geo_df['region'].unique().tolist()

state_list = geo_df['state'].unique().tolist()

metro_area_list = geo_df['metro_area'].unique().tolist()

county_list = geo_df['county_st'].unique().tolist()

#compare_input = st.sidebar.radio('choose a comparison', options=['largest change','closest places', 'similar places', 'random'])

# graphs
if view_input == 'region':
    region_input = st.sidebar.multiselect('Choose Region(s):',options=region_list)
    options = ['cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'land_sqmi_region', 'population_region', 'pop_per_sqmi_region',
    'pct_white_region', 'pct_nonwhite_region', 'pct_insured_region', 'pct_uninsured_region', 'region_lat', 'region_lon']

    no_scatter = st.sidebar.number_input(f'{view_input}s visible on heatmap & scatterplot', value=10, max_value=12)
    st.sidebar.subheader('Choose scatterplot elements')
    scatter_x = st.sidebar.selectbox('x axis', options=options, index=6)
    scatter_y = st.sidebar.selectbox('y axis', options=options, index=0)
    scatter_sz = st.sidebar.selectbox('bubble size', options=options, index=5)
    scatter_col = st.sidebar.selectbox('color', options=options, index=1)

    region_map, region_bar, region_heatmap, region_scatter = region_grapher(df)
    subtitle_map = f'Regional change in {metric_input} over past {days_input} days'
    st.subheader(subtitle_map)
    st.plotly_chart(region_map)

    subtitle_bar = f'Change in {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_bar)
    st.plotly_chart(region_bar)

    subtitle_hm = f'Tracking {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_hm)
    st.plotly_chart(region_heatmap)

    subtitle_scatter = f'Comparison of {no_scatter} {view_input}s by {scatter_x} & {scatter_y}'
    st.subheader(subtitle_scatter)
    st.plotly_chart(region_scatter)

elif view_input == 'state':
    state_input = st.sidebar.multiselect('Choose State(s):',options=state_list)
    options = ['cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'land_sqmi_state', 'population_state', 'pop_per_sqmi_state',
    'pct_white_state', 'pct_nonwhite_state', 'pct_insured_state', 'pct_uninsured_state', 'state_lat', 'state_lon']

    no_scatter = st.sidebar.number_input(f'{view_input}s visible on heatmap & scatterplot', value=7, max_value=51)
    st.sidebar.subheader('Choose scatterplot elements')
    scatter_x = st.sidebar.selectbox('x axis', options=options, index=6)
    scatter_y = st.sidebar.selectbox('y axis', options=options, index=0)
    scatter_sz = st.sidebar.selectbox('bubble size', options=options, index=5)
    scatter_col = st.sidebar.selectbox('color', options=options, index=1)

    state_map, state_bar, state_heatmap, state_scatter = state_grapher(df)
    subtitle_map = f'State change in {metric_input} over past {days_input} days'
    st.subheader(subtitle_map)
    st.plotly_chart(state_map)

    subtitle_bar = f'Change in {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_bar)
    st.plotly_chart(state_bar)

    subtitle_hm = f'Tracking {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_hm)
    st.plotly_chart(state_heatmap)

    subtitle_scatter = f'Comparison of {no_scatter} {view_input}s by {scatter_x} & {scatter_y}'
    st.subheader(subtitle_scatter)
    st.plotly_chart(state_scatter)

elif view_input == 'metro':
    metro_input = st.sidebar.multiselect('Choose Metro Area(s):',options=metro_area_list)
    options = ['cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'land_sqmi_metro_area', 'population_metro_area', 'pop_per_sqmi_metro_area',
    'pct_white_metro_area', 'pct_nonwhite_metro_area', 'pct_insured_metro_area', 'pct_uninsured_metro_area', 'metro_area_lat', 'metro_area_lon']

    no_scatter = st.sidebar.number_input(f'{view_input}s visible on heatmap & scatterplot', value=12, max_value=500)
    st.sidebar.subheader('Choose scatterplot elements')
    scatter_x = st.sidebar.selectbox('x axis', options=options, index=6)
    scatter_y = st.sidebar.selectbox('y axis', options=options, index=0)
    scatter_sz = st.sidebar.selectbox('bubble size', options=options, index=5)
    scatter_col = st.sidebar.selectbox('color', options=options, index=1)

    metro_map, metro_bar, metro_heatmap, metro_scatter = metro_grapher(df)
    subtitle_map = f'Metro Area change in {metric_input} over past {days_input} days'
    st.subheader(subtitle_map)
    st.plotly_chart(metro_map)

    subtitle_bar = f'Change in {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_bar)
    st.plotly_chart(metro_bar)

    subtitle_hm = f'Tracking {metric_input} every {days_input} days for selected {view_input}s'
    st.subheader(subtitle_hm)
    st.plotly_chart(metro_heatmap)

    subtitle_scatter = f'Comparison of {no_scatter} {view_input}s by {scatter_x} & {scatter_y}'
    st.subheader(subtitle_scatter)
    st.plotly_chart(metro_scatter)

else:
    county_input = st.sidebar.multiselect('Choose county(ies):',options=county_list)
    options = ['cases', 'cases_per_mil', 'deaths', 'deaths_per_mil', 'land_sqmi_county', 'population_county', 'pop_per_sqmi_county',
    'pct_white_county', 'pct_nonwhite_county', 'pct_insured_county', 'pct_uninsured_county', 'county_lat', 'county_lon']

    no_scatter = st.sidebar.number_input('counties visible on heatmap & scatter plot', value=12, max_value=500)
    st.sidebar.subheader('Choose scatterplot elements')
    scatter_x = st.sidebar.selectbox('x axis', options=options, index=6)
    scatter_y = st.sidebar.selectbox('y axis', options=options, index=0)
    scatter_sz = st.sidebar.selectbox('bubble size', options=options, index=5)
    scatter_col = st.sidebar.selectbox('color', options=options, index=1)

    county_map, county_bar, county_heatmap, county_scatter = county_grapher(df)
    subtitle_map = f'County change in {metric_input} over past {days_input} days'
    st.subheader(subtitle_map)
    st.plotly_chart(county_map)

    subtitle_bar = f'Change in {metric_input} every {days_input} days for selected counties'
    st.subheader(subtitle_bar)
    st.plotly_chart(county_bar)

    subtitle_hm = f'Tracking {metric_input} every {days_input} days for selected counties'
    st.subheader(subtitle_hm)
    st.plotly_chart(county_heatmap)

    subtitle_scatter = f'Comparison of {no_scatter} counties by {scatter_x} & {scatter_y}'
    st.subheader(subtitle_scatter)
    st.plotly_chart(county_scatter)
