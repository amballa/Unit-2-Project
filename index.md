![image](https://user-images.githubusercontent.com/92558174/146867019-88381d28-9055-49bb-9345-4b0994e10052.png)
_Point guard Ja Morant, drafted 2nd overall by the Memphis Grizzlies in 2019_

### Background
The NBA is amongst the most popular and premier sports leagues in the world. The league truly represents the best of the best in the world of basketball. But before getting a chance to play on the biggest stage in basketball, prospective players must prove themselves on a smaller stage - whether in the minor leagues, internationally, or more commonly in colleges across the country. 

Using the performance statistics of college basketball players, my goal is to predict which althletes will be drafted by NBA teams in a given year.


### Dataset

Of the 65 columns in the dataset, I chose to make use of 26 in training the predictive model.


### Cleaning and Wrangling Preparation
Here is my wrangle function:

```
def wrangle(filepath):
  df = pd.read_csv('/content/CollegeBasketballPlayers2009-2021.csv')

  df = df[df['year'] >= 2016].reset_index(drop=True)

  col_drop = ['DRB_per', 'AST_per', 'FTM', 'FTA', 'twoPM', 'ast/tov', 'adrtg', \
              'twoPA', 'TPM', 'TPA', 'blk_per', 'stl_per', 'num', 'ht',\
              'porpag', 'adjoe', 'pfr', 'type', 'Rec Rank', 'rimmade', \
              'rimmade+rimmiss', 'midmade', 'midmade+midmiss', 'rimmade/(rimmade+rimmiss)', \
              'midmade/(midmade+midmiss)', 'dunksmiss+dunksmade', \
              'dunksmade/(dunksmade+dunksmiss)', 'dporpag', 'obpm', \
              'dbpm', 'gbpm', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'Unnamed: 65']
  #col_drop.append(['eFG', 'TS_per'])
  df.drop(columns=col_drop, inplace=True)
  #df.columns = []
  
  df.rename(columns = {'Unnamed: 64': 'pos', 'dunksmade': 'dunks'}, inplace=True)
  #df.drop(columns = ['pos'], inplace=True)

  # Imputing NaN values
  df['dunks'].fillna(0, inplace=True)

  #Removing outlier observations
  df.drop(index = df[df['Ortg'] > 145.96].index, inplace=True)
  df.drop(index = df[df['eFG'] > 81.3].index, inplace=True)
  df.drop(index = df[df['TS_per'] > 77.8].index, inplace=True)
  df.drop(index = df[df['ORB_per'] > 27.6].index, inplace=True)
  df.drop(index = df[df['ftr'] > 150].index, inplace=True)
  df.drop(index = df[df['drtg'] < 0].index, inplace=True)
  df.drop(index = df[df['bpm'] > 25].index, inplace=True)
  df.drop(index = df[df['mp'] > 40].index, inplace=True)

  #Some feature engineering 
  def set_major(conf):
    high_major = ['B10', 'SEC', 'ACC', 'B12', 'P12', 'BE']
    mid_major = ['Amer', 'A10', 'MWC', 'WCC', 'MVC', 'CUSA', 'MAC']
    if conf in high_major:
      return 'high'
    elif conf in mid_major:
      return 'mid'
    else:
      return 'low'
  df['conf_mjr'] = df['conf'].apply(set_major)

  # Creating the target column
  df['drafted'] = df['pick'].notnull().astype(int)
  df.drop(columns='pick', inplace=True)
  df.sort_values(by='year', ascending=False, inplace=True)
  draft_picks = df[['pid', 'drafted']][df['drafted'] == 1]
  late_draft_ind = df.loc[draft_picks[draft_picks.duplicated()].index].index
  df.loc[late_draft_ind, 'drafted'] = 0

  # Dropping high-cardinality team column
  df.drop(columns=['team'], inplace=True)

  # Dropping rows with any null value
  df.dropna(axis=0, how='any', inplace=True)
  df.drop(index = df[df['yr'] == 'None'].index, axis =0, inplace=True)

  df.drop(columns = ['pid'], inplace=True)
  df.set_index('player_name', inplace=True)
  
  return df
```

(image here)

```
(code here)
```


```
(code here
```

(image here)

```
(code here)
```

```markdown
(code here)
```


```markdown
(code here)
```

(image here)


### Logistic Regression Classification

```
(some code maybe)
```
(some graphs maybe)


### Tree Model Classification
```
(some code maybe)
```
(some image or graph maybe)

blah blah blah

### Boosted Tree Model Classification

```
(some code maybe)
```

### Initial Model Comparison

(whatever)


### Hyperparameter Tuning and Final Model Comparison

### Final Prediction



### Concluding Thoughts and Analysis Limitations

