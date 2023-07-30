from datetime import datetime
from datetime import timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
stockDf = CreatePdDataframeForSingleStockPrice()
stockDf.reset_index(inplace=True)
stockClose = stockDf[['Date','Close']]
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2010-1-1', end='2023-7-1')
tweets = pd.read_csv('TweetsElonMusk_Date.csv')
label = pd.DataFrame(columns=['Label'])

# Group all tweets on a given day into one
tweets = tweets.groupby('Date').agg(lambda x: ';'.join(x))

# Combine Saturday & Sunday into Monday
tweets.reset_index(inplace=True)
for index, row in tweets.iterrows():
  today_str = row[0]
  today_dt = datetime.strptime(today_str, '%Y-%m-%d')
  tom_dt = today_dt + timedelta(days = 1)
  next_dt = today_dt + timedelta(days = 2)
  if today_dt.weekday() == 6:
    tweets.loc[index,'Date'] = tom_dt.strftime('%Y-%m-%d')
  if today_dt.weekday() == 5:
    tweets.loc[index,'Date'] = next_dt.strftime('%Y-%m-%d')
tweets = tweets.groupby('Date').agg(lambda x: ';'.join(x))

# Combine holiday with day after
tweets.reset_index(inplace=True)
for index, row in tweets.iterrows():
  if holidays.isin([row[0]]).any().any():
    tweets.loc[index,'Date'] = tweets.loc[index+1,'Date']
tweets = tweets.groupby('Date').agg(lambda x: ';'.join(x))

# Change stock data from datetime to date
for index, row in stockClose.iterrows():
  dt = row['Date']
  stockClose.iloc[index,0] = dt.date()

# Create label for each "day" of tweets
for index, row in tweets.iterrows():
  # Find current, previous, and next calendar dates
  today_dt = datetime.strptime(index, '%Y-%m-%d')
  today = today_dt.date()
  yest_dt = today_dt - timedelta(days = 1)
  yesterday = yest_dt.date()
  tom_dt = today_dt + timedelta(days = 1)
  tomorrow = tom_dt.date()

  # If stock data exists calculate label
  if stockClose.isin([today]).any().any():
    today_index = stockClose.index[stockClose.Date == today].values
    prev2_close = stockClose.iloc[today_index-2].values[0,1]
    prev_close = stockClose.iloc[today_index-1].values[0,1]
    next_close = stockClose.iloc[today_index+1].values[0,1]
    prev_growth = (prev_close - prev2_close)/prev2_close
    curr_growth = (next_close - prev_close)/prev_close
    growth_delta = curr_growth - prev_growth
    if growth_delta > 0:
      label.loc[len(label)] = {'Label':1}
    else:
      label.loc[len(label)] = {'Label':0}
  else:
    label.loc[len(label)] = {'Label':None}

# Combine tweets and labels
tweets.reset_index(inplace=True)
dataset = pd.concat([tweets.Tweet, label], axis=1)

# Remove unlabeled tweets
dataset = dataset.dropna()
