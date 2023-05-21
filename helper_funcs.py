# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:04:29 2023

@author: yaobv
"""

import pandas as pd
import numpy as np
import math
from vincenty import vincenty
from sklearn.cluster import DBSCAN

def wrangle(port_file=None, tracking_file=None):
    
    # loading the ports data
    df1 = pd.read_csv(port_file)

    # converting lat and long to radians to compute haversine distance
    df1['lat_rad'] = np.radians(df1['lat'])
    df1['long_rad'] = np.radians(df1['long'])

    # rounding lat and long in port df
    df1['lat'] = df1['lat']
    df1['long'] = df1['long']
    df1['lat_long'] = [[x, y] for x, y in zip(df1['lat'], df1['long'])]
    
    # loading the tracking data
    df2 = pd.read_csv(tracking_file, parse_dates=['datetime'])
    df2 = df2.drop_duplicates()
    df2 = df2.sort_values(['vessel', 'datetime'])
    df2['vessel_1back'] = df2['vessel'].shift()

    # converting lat and long to radians to compute haversine distance
    df2['lat_rad'] = np.radians(df2['lat'])
    df2['long_rad'] = np.radians(df2['long'])

    # adding lat/long column and lat/long 1 back to later compute delta
    df2['lat_long'] = [[x, y] for x, y in zip(df2['lat'], df2['long'])]
    df2['lat_long_1back'] = df2.groupby('vessel')['lat_long'].transform(lambda x: x.shift())
    df2['direction'] = pd.cut(df2['heading'], bins=[0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360],
                              labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'],
                              ordered=False, include_lowest=True)

    # time delta and hour delta that will be used for filtering - i do it again
    # after the filter
    df2['time_delta'] = df2.groupby('vessel')['datetime'].transform(lambda x: x - x.shift())
    df2['hour_delta'] = [n.total_seconds() / 3600 for n in df2['time_delta']]

    # i divide the map into quadrants to calculate the proportion of the time each
    # vessel was in each quadrant
    conditions = [((df2['lat'] > 0) & (df2['long'] > 0)),
                  ((df2['lat'] < 0) & (df2['long'] > 0)),
                  ((df2['lat'] > 0) & (df2['long'] < 0)),
                  ((df2['lat'] < 0) & (df2['long'] < 0))
                  ]
    labels = ['quad1', 'quad4', 'quad2', 'quad3']
    df2['quad'] = np.select(conditions, labels)

    # filtering using query to eliminate unneeded/impossible values
    df2 = df2.query('speed < 30 & heading <= 360 & draft < 13.5')
    df2 = df2.reset_index(drop=True)

    return df1, df2


def vincenty_distance(row):
  """ 
  returns the vincenty distance for contiguous rows - will be used to identify
  impossible distances travelled
  """
  if row['vessel'] != row['vessel_1back']:
    return -99

  loc1 = row['lat_long']
  loc2 = row['lat_long_1back']

  try:
    distance = vincenty(loc1, loc2)
    return distance
  except:
    return 


def nearest_port(row, kneigh_model, idx_ports, radius):
  """
  returns the port identifier of the nearest port using the nearest neighbors
  model 
  """

  data = np.array([row['lat_rad'], row['long_rad']]).reshape(1, -1)
  dist, pred = kneigh_model.radius_neighbors(
      data, radius=radius, sort_results=True)

  if len(dist[0]) == 0:
    return -1

  else:
    return idx_ports[pred[0][0]]


def nearest_distance(df, kneigh_model=None, radius=0.015):
  """
  returns the distance of the nearest port in the dataset
  """

  data = np.array([df['lat_rad'], df['long_rad']]).reshape(1, -1)
  dist, pred = kneigh_model.radius_neighbors(
      data, radius=radius, sort_results=True)

  if len(dist[0]) == 0:
    return -1

  else:
    return dist[0][0]

def vincenty_port(row):
    """
    a function that computes the vincenty distance between the assigned port
    and the latitude and longitude of the location data
    """
    port = row['port_coords']
    loc = row['lat_long']
    return vincenty(port, loc)


def fix_ports(df):
    """
    fills out port sequences if there's a draft change in the sequence. this will
    help get better arrival and departure dates. it should be applied to 
    groupby DataFrames 
    
    parameters
    ----------
        df: a pandas DataFrame
    
    returns
    -------
        df: a pandas Series
    """
    
    ugh = []

    for x in df['port_sequence'].unique():
        df_ = df.query(f'port_sequence == {x}')
        ports = df_['pred_port']
        new_ports = []
      
        if not all(ports <= 0):
            new_ports.append([ports.max()] * len(ports))
        else:
            new_ports.append(ports)
    
        ugh.append(new_ports[0])

    thisthis = []
    
    for n in ugh:
      for p in n:
        thisthis.append(p)
    
    return thisthis


def get_voyages(df):
    """
    converts the port sequences in each dataframe into voyages
    with the proper formatting
    
    param:
    -----
        df: pandas DataFrame
    
    returns:
    -------
        df: processed pandas DataFrame
    
    """
    # filtering out rows without an assigned port
    
    nz = df[(df['pred_port'] > 0) | (df['pred_port'] == -75)].reset_index()
    
    vessel = nz['vessel'][0]
    dt = nz['datetime']
    pred = nz['pred_port']
    
    records = []
    
    for i in range(len(dt)-1):
        if pred[i] != pred[i+1]:
            start_port = pred[i]
            end_port = pred[i+1]
            begin_date = dt[i]
            end_date = dt[i+1]
            records.append([vessel, begin_date, end_date, start_port, end_port])

    df = (pd.DataFrame.from_records(records,
                      columns=[
                          'vessel', 'begin_date', 'end_date', 'begin_port_id', 'end_port_id'])
          )

    return df


def get_angle(loc1=[1.2, 103], loc2=[99.5, -32], kmeans_labels=None):
  """
  get the angle of the bearing needed to directly approach one location from
  another

  parameters
  ----------
        lists: lat, longitude of ship (in dict as such)

  returns
  -------
        float: bearing in degrees with 0 as due North
  """

  dLon = (loc2[1] - loc1[1])

  y = math.sin(dLon) * math.cos(loc2[0])
  x = math.cos(loc1[0]) * math.sin(loc2[0]) - \
      math.sin(loc1[0]) * math.cos(loc2[0]) * math.cos(dLon)

  brng = math.atan2(y, x)

  brng = math.degrees(brng)
  brng = (brng + 360) % 360
  brng = 360 - brng  # count degrees clockwise - remove to make counter-clockwise

  return brng


def prepare_data(df=None, n_input=3, kmeans_label=None):
    """
    preparing the sequences for window based models
    """
    
    df = get_voyages(df)
    vessel = df['vessel'].iloc[0]
    ports_ = np.array(
        pd.concat([df['begin_port_id'],
                   pd.Series(df['end_port_id'].iloc[-1])]))
    
    X = []
    Y = []
    start = 0

    for i in range(len(ports_)):
      last_input = start + n_input
      last_output = last_input + 3
      if last_output <= len(ports_):
        x = ports_[start:last_input]
        y = ports_[last_input: last_output]
        X.append(x)
        Y.append(y)
        start += 1
    try:
      df = pd.concat([pd.DataFrame(X),
                      pd.DataFrame(Y, columns=['port_1ahead', 'port_2ahead', 'port_3ahead'])], axis=1)

    except:
      df = pd.DataFrame()

    # X = []
    
    # for x in X:
    #   for n in x:
    #     if n == -75:
    #       port_coords = [33, 140]
    #     else:
    #       port_coords = list(ports[n])
    #     port = [n]
    #     port.extend(port_coords)
    #     new_X.append(port)
    
    df['vessel'] = len(df) * [vessel]
    df['kmeans_label'] = [kmeans_label[n] for n in df['vessel']]
    
    return df.astype(int)

def train_dbscan(df=None, eps=0.1, min_samples=4):
    """
    use the dbscan clustering algorithm to find groups of ports.
    chose to use it and think it over, but also because it can find 
    irregularly shaped clusters without specifying their number in advance.
    
    params
    ------
        df: pandas df
        eps: min distance between points in cluster
        min_samples: min members in a cluster
    
    returns
    -------
        labels: labels matching index of long/lat input pairs
    """
    
    coords = df[['long_rad', 'lat_rad']].values
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    db.fit(coords)
    
    return db.labels_

def get_pred_data(df=None, n_input=3):
    """
    preparing the submission data for prediction by getting the port
    sequences
    """
    
    df = get_voyages(df)
    ports_ = np.array(pd.concat([df['begin_port_id'], (pd.Series(df['end_port_id'].iloc[-1]))]))
    
    pred_seq = ports_[-n_input:]

    if len(pred_seq) < n_input:
        pred_seq = np.insert(pred_seq, 0, pred_seq[-1])

    return pred_seq

def get_prior_port(df):
    """
    iterate through each vessel's predicted ports to get the last predicted
    port
    """
    
    p = df['pred_port'][::-1].values
    
    prior_ports = []

    for i in range(len(p)):
        s = p[i]        
        for n in p[i:]:            
            if (s != n) & (n >0) :
                prior_ports.append(n)
                break

    # padding zeros at the end to indicate no next port 
    zeroes = [0] * (len(p) - len(prior_ports))
    prior_ports = prior_ports + zeroes
    
    prior_ports = prior_ports[::-1]
    
    return prior_ports

def fix_close_ports(df):
    """
    compares adjacent port assignments minimum distances and returns a new series
    with ports reassigned where appropriate
    
    parameters
    ----------
        df: a pandas DataFrame
    
    returns
    -------
        df: a pandas Series
    """ 
    
    new_destinations = []

    for seq in range(1, len(df['port_sequence'].unique()) +1):
        seq1_port = df[df['port_sequence'] == seq].pred_port
        seq2_port = df[df['port_sequence'] == seq + 1].pred_port
        seq1_mindist = df[df['port_sequence'] == seq].pred_port_dist
        seq2_mindist = df[df['port_sequence'] == seq+1].pred_port_dist
        new_ports = []

        if (seq1_port.max() <= 0) or (seq2_port.max() <= 0):
            new_ports.append([seq1_port.max()] * len(seq1_port))
    
        elif seq2_mindist.min() < seq1_mindist.min():
            true_port = seq2_port
            new_ports.append([true_port.max()] * len(seq1_port))
        else:
            new_ports.append([seq1_port.max()] * len(seq1_port))
    
        new_destinations.append(new_ports[0])
  
    new_dest = []

    for dest in new_destinations:
        for d in dest:
            new_dest.append(d)

    return new_dest

def fix_really_close_cluster(row):
    """
    Applies filters to ports: come back to this. It's very hacky.

    """
  
    duration = row['port_sequence_time']
    dist = row['pred_port_dist']

    if row['pred_port_backup'] in ([72, 152]):
        if ((duration > 48) & (dist < 50) & (row['heading_sequence_time'] > 8)):
            return row['pred_port_backup']
        elif dist < 5:
            return row['pred_port_backup']
        elif duration > 100:
            return row['pred_port_backup']
        else:
            return row['pred_port']

    elif row['pred_port_backup'] == 99:
        if ((dist >20) & (row['draft_change'] < 1)):
            return 0
        else:
            return row['pred_port']

    elif row['pred_port_backup'] in ([115, 54]):
        if ((dist < 15) & (row['heading_sequence_time'] > 12)):
            return row['pred_port_backup']
        else:
            return 0

    elif dist < 2:
        if ((duration > 16) & (dist <2)):
            return row['pred_port_backup']
        else:
            return row['pred_port']

    elif row['pred_port'] == 21:
        if dist > 25:
            return 0
        else:
            return row['pred_port']

    elif row['heading_sequence_time'] > 30:
        if dist <= 10:
            return row['pred_port_backup']
        else:
            return row['pred_port']
  
    elif row['heading_sequence_time'] > 6:
        if ((duration >25) & (dist <=15)):
            return row['pred_port_backup']
        else:
            return row['pred_port']

    else:
        return row['pred_port']
