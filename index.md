![image](https://user-images.githubusercontent.com/92558174/146867019-88381d28-9055-49bb-9345-4b0994e10052.png)
_NBA commissioner Adam Silver congratulating point guard Ja Morant, drafted 2nd overall in 2019_

## Background
The NBA is amongst the most popular and premier sports leagues in the world. Attracting millions of viewers and generating billions in annual revenue, the league truly represents the best of the best in the world of basketball. But the route to the NBA is difficult to say the least. Before getting a chance to play on the biggest stage in basketball, prospective players must prove themselves on smaller stages - whether in the American minor league, internationally, or most often on college courts. Every year, 60 individuals are chosen to join the ranks of the pros in the annual NBA draft. Around 50 or so are drafted directly from college. This selection comes from a pool of over 4,500 Division 1 players across 350 teams. That's a rate of 1.1%! Who are these select few and why are they chosen?

Using on-court stats, my goal is to predict which NCAA players will be drafted by NBA teams in a given year and find the stats that best predict draft success.

## Dataset

The [College Basketball + NBA Advanced Stats](https://www.kaggle.com/adityak2003/college-basketball-players-20092021) dataset on Kaggle contains stats for NCAA Division 1 basketball players from 2009 to 2021. It contains over 60,000 rows representing the seasons of over 25,000 players. Of the 65 columns in the dataset, I chose to make use of 25 and engineered one additional feature: conf_mjr.

##### Categorical:
> **conf**: Conference
> 
> **conf_mjr**: Conference tier
> 
> **pos**: Position
> 
> **yr**: Year of college
> 

##### Numeric:
> **GP**: Games played
> 
> **Min_per**: Minutes %
> 
> **Ortg**: Offensive rating
> 
> **drtg**: Defensive rating
> 
> **usg**: Usage rate
> 
> **eFG**: Effective field goal %
> 
> **TS_per**: True shooting %
> 
> **FT_per**: Free throw %
> 
> **twoP_per**: 2-pointer %
> 
> **TP_per**: 3-pointer %
> 
> **ftr**: Free throw rate
> 
> **TO_per**: Turnover %
> 
> **ORB_per**: Offensive rebound %
> 
> **treb**: Average rebounds
> 
> **dunks**: Dunks made
> 
> **stops**: Stops made
> 
> **bpm**: Box plus/minus
> 
> **mp**: Average minutes played
> 
> **ast**: Average assists
> 
> **stl**: Average steals
> 
> **blk**: Average blocks
> 
> **pts**: Average points



## Preparation for Model Building
### Cleaning and Wrangling
Although the dataset was well organized and contained relatively few null values, it required some basic wrangling to reduce the size and deal with missing data and outlier observations. I decided on retaining data from the 2015-2016 season through the 2020-2021 season and removed the columns of all the stats I was not interested in. Given the cardinality of the _team_ feature (362), I decided to drop that column as well. To make things interesting, I also created an additional feature - _conf_mjr_ which rates the competitiveness of the conference an indvidual plays in as either high, mid, or low. For the target, I used the _pick_ feature to created a binary column representing draft status. The full wrangle function is included below for reference.

```
def wrangle(filepath):
  df = pd.read_csv('/content/CollegeBasketballPlayers2009-2021.csv')

  df = df[df['year'] >= 2016].reset_index(drop=True)

  col_drop = ['DRB_per', 'AST_per', 'FTM', 'FTA', 'twoPM', 'ast/tov', 'adrtg', 
              'twoPA', 'TPM', 'TPA', 'blk_per', 'stl_per', 'num', 'ht', 'porpag',
              'adjoe', 'pfr', 'type', 'Rec Rank', 'rimmade', 'midmade',
              'rimmade+rimmiss', 'midmade+midmiss', 'rimmade/(rimmade+rimmiss)', 
              'midmade/(midmade+midmiss)', 'dunksmiss+dunksmade', 
              'dunksmade/(dunksmade+dunksmiss)', 'dporpag', 'obpm', 
              'dbpm', 'gbpm', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'Unnamed: 65']
  df.drop(columns=col_drop, inplace=True)
  
  df.rename(columns = {'Unnamed: 64': 'pos', 'dunksmade': 'dunks'}, inplace=True)

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

### Train-Validation-Test Split


```
cutoff = 2020
df_train = df[df['year'] < cutoff]
df_val = df[df['year'] == cutoff]
df_test = df[df['year'] > cutoff]
```

### Feature Matrix and Target Array

```
X_train = df_train.drop(columns = [target, 'year'])
y_train = df_train[target]

X_val = df_val.drop(columns = [target, 'year'])
y_val = df_val[target]

X_test = df_test.drop(columns = [target, 'year'])
y_test = df_test[target]
```

### Baseline

```
baseline_train_acc = y_train.value_counts(normalize=True).max()*100
baseline_val_acc = y_val.value_counts(normalize=True).max()*100
```

## Classification Models
### Logistic Regression

```
model_log = make_pipeline(OneHotEncoder(use_cat_names = True),
                          StandardScaler(),
                          LogisticRegression(n_jobs=-1, random_state=42)
                          )
model_log.fit(X_train, y_train)
```
....


### Tree Models
```
model_ada = make_pipeline(OrdinalEncoder(),
                          AdaBoostClassifier(random_state=42)
                          )
model_ada.fit(X_train, y_train);


model_xgb = make_pipeline(OrdinalEncoder(),
                          XGBClassifier(random_state=41, n_jobs=-1)
                          )
model_xgb.fit(X_train, y_train);
```

....


### Initial Model Comparison
(whatever)

![image](https://user-images.githubusercontent.com/92558174/147169110-851030dd-5c67-475d-9338-12fd9ec18d47.png)

![image](https://user-images.githubusercontent.com/92558174/147194006-08c0f22f-c158-4bfb-8b41-2224c28e122c.png)



### Hyperparameter Tuning and Final Model Comparison

```
model_log = make_pipeline(OneHotEncoder(use_cat_names = True),
                          StandardScaler(),
                          LogisticRegression(n_jobs=-1, random_state = 42)
                          )
param_grid = {'logisticregression__max_iter': [100, 200, 300],
              'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'logisticregression__penalty': ['none', 'l1', 'l2', 'elasticnet'],
              'logisticregression__C': [10, 1.0, 0.1]
              }
model_log_s = GridSearchCV(model_log,
                          param_grid=param_grid,
                          n_jobs=-1,
                          cv=5,
                          scoring='f1',
                          verbose=1
                          )
model_log_s.fit(X_train, y_train)
```

```
model_ada = make_pipeline(OrdinalEncoder(),
                          AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
                          )
param_grid = {'adaboostclassifier__base_estimator__max_depth': [1,2],
              'adaboostclassifier__learning_rate': [0.1, 0.5, 1],
              'adaboostclassifier__n_estimators': [50,100,150]
              }

model_ada_s = GridSearchCV(model_ada,
                           param_grid=param_grid,
                           n_jobs=-1,
                           cv=5,
                           scoring='f1',
                           verbose=1,
                            )
model_ada_s.fit(X_train, y_train)
```

```
model_xgb = make_pipeline(OrdinalEncoder(),
                          XGBClassifier(n_jobs=-1, random_state = 42)
                          )
param_grid = {'xgbclassifier__scale_pos_weight': [1, 10, 50, 99],
              'xgbclassifier__learning_rate': [0.01, 0.1, 0.3],
              'xgbclassifier__max_depth': [3,6,9],
              'xgbclassifier__n_estimators': [50,100,150]
              }
model_xgb_s = GridSearchCV(model_xgb,
                           param_grid=param_grid,
                           n_jobs=-1,
                           cv=5,
                           scoring='f1',
                           verbose=1
                           )
model_xgb_s.fit(X_train, y_train)
```

| Tuned Model         | Accuracy  | ROC AUC  | Precision | Recall | F1 Score |
|:--------------------|:----------|----------|:----------|:-------|:---------|
| Logistic Regression | 99.358974 | 0.996393 | 0.83      | 0.51   | **0.63** |
| Adaptive Boost      | 99.204244 | 0.981694 | 0.70      | 0.47   | **0.56** |
| XGBoost             | 99.005305 | 0.993492 | 0.53      | 0.71   | **0.61** |


![image](https://user-images.githubusercontent.com/92558174/147180713-d02659a5-5ebf-42cc-8252-80c1cb8f1728.png)


### Final Predictions for 2021 Draft

|![image](https://user-images.githubusercontent.com/92558174/147177924-f74402e5-da06-448d-84e6-b1b08dd7ac0b.png) | ![image](https://user-images.githubusercontent.com/92558174/147183927-ca56f4a2-7d88-454f-8a3b-50039b2248f4.png)|


Seeing that the logistic regression model did not perfrom all that well on the test set, I decided to test the updated XGBoost as well. Here are the results:

|![image](https://user-images.githubusercontent.com/92558174/147178013-2e0960b7-1a08-43a6-9e63-d4955066f342.png) | ![image](https://user-images.githubusercontent.com/92558174/147176443-a181952f-89e1-4ddd-b94b-c1ecebf5e3be.png)|

### Important Features
Permutation importances for the tuned LR model:
![image](https://user-images.githubusercontent.com/92558174/147185241-007b2268-8574-439d-ac39-39edd4199617.png)

Permutation importances for the tuned XGBoost model:
![image](https://user-images.githubusercontent.com/92558174/147177421-a5b66a2a-8cd4-4337-9756-b9c2c079b40c.png)

To my surprise, the two models "cared" about very different features!

## Concluding Thoughts
My approach is very basic.
Some potential improvements could be made by:
* finding the right combination of stats
* handling 0s and 1s for percentage features better or dropping those observations altogether
* adding a feature to distinguish draft-eligible players 
* adding a feature to distinguish players who have actually declared for the draft
* adding or calculating a composite score feature
* ulitizing an undersamping or oversampling technique like SMOTE

It would also be interesting to see how well a more complex model could predict not only which players will drafted, but the order in which they are chosen.
