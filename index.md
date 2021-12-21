---
title: Historical Relationship between UFO Sightings and Popularity of Science-Fiction Movies
---
![image](https://user-images.githubusercontent.com/92558174/142338159-5916511c-a2ef-414b-89ae-93ef7422102a.png) _Forbidden Planet (1956)_

### Background
Humans have long been fascinated by space and the prospect of intelligent extraterrestrial life. Surely, the cosmos is too vast for a bipedal ape species to be the only technologically-inclined life-form! We look to the sky for answers with equal parts hope and fear. But what we see is often not so clear. Whether it's blinking lights or impossible aerial movements, many have claimed to lay sight on phenomena beyond explanation. Over the past century alone, the National UFO Reporting Center has catalogued over 80 thousand eyewitness accounts of unidentified flying objects across the world. Perhaps unsurprisingly, nearly 90% of those accounts came from Americans. So what is it about America that makes UFO sightings such a regular occurence?  Is there really something non-human out there or are the sightings the result of an overactive collective imagination? Are there cultural or political forces at play that might explain why so many Americans have these strange experiences? These questions remain open. But with this analysis, my goal is to compare the trends in the number of UFO sightings in the US with the box office performance of science-fiction films. I will also account for changes in the country's population over the same time period.



### Datasets
For this analysis, I used three publicly available datasets:
1. UFO Sightings (National UFO Reporting Center via Kaggle)
    - 80,332 records from 1910 to 2014 with date, location, coordinates, shape of object, duration, and eyewitness comments.

2. TMDB 5000 Movie (The Movie Database via Kaggle)
    - 10,866 movie titles from 1960 to 2015 with information such as release date, cast, genre, plot summary, IMDB rating, production budget, and box office revenue .

3. US Population by Year (US Census Bureau)
    - Resident population of the United States from 1900 to 2020.
    

Limited by the time frames of the NUFORC and TMDB datasets, I chose to compare data from 1960 to 2013. Additionally, because the UFO reports date back to the early 20th century - long before the establishment of the reporting center - older eyewitness accounts may be obscured and under-reported.

### Hypotheses
I will be performing two linear regressions: one with a single variable and the other with two variables.
- Dependent variable = number of UFO sightings in a given year
- Independent variable #1 = revenue of movies in the Sci-Fi genre in a given year
- Independent variable #2 = US population in a given year

For the **single variable model**:
- Null Hypothesis: there is no relationship between the number of UFO sightings and revenue of Sci-Fi movies
- Alternative Hypothesis: there _is_ a relationship between the variables

For the **multi-variable model**:
- Null Hypothesis: there is no relationship between the number of UFO sightings and revenue of Sci-Fi movies after accounting for population
- Alternative Hypothesis: there _is_ a relationship between the variables, even after accounting for population


### Data Wrangling & Feature Engineering
Both the UFO and movie datasets required quite a bit of data manipulation and cleanup. With the UFO dataset, the first step was to remove all entries from outside the United States. For this, I used the for loop in the code below since a significant number of entries contained the state abbreviation but nothing under the country header. Then to extract the year, I needed to convert the dates in strings into Pandas datetime objects. But because the Pandas to_datetime function cannot parse a time of 24:00 (midnight), I created a function to change each 24:00 in the date string to 0:00. From there, it was a straightforward sorting chronologically and creating a frequency table of the number of sightings per year.

```
ufo = pd.read_csv('scrubbed.csv')
ufo.head()
```

![image](https://user-images.githubusercontent.com/92558174/141884964-9e195c29-0d9b-42d9-a557-ce6fa71b4138.png)

```
states = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 
'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 
'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 
'wa', 'wv', 'wi', 'wy']

for i in range(len(ufo)):
  if ufo.loc[i]['state'] in states or ufo.loc[i]['country'] == 'us':
    pass
  else:
    ufo.drop(index = i, inplace=True)

def fix_datetime_format(date_str):
  if date_str[11:13] == '24' and date_str[2] == '/' and date_str[5] == '/':
    date_str = date_str[0:11] + '00' + date_str[13:]
    return date_str
  elif date_str[10:12] == '24' and date_str[2] == '/' and date_str[4] == '/':
    date_str = date_str[0:10] + '00' + date_str[12:]
    return date_str
  elif date_str[10:12] == '24' and date_str[1] == '/' and date_str[4] == '/':
    date_str = date_str[0:10] + '00' + date_str[12:]
    return date_str
  elif date_str[9:11] == '24' and date_str[1] == '/' and date_str[3] == '/':
    date_str = date_str[0:9] + '00' + date_str[11:]
    return date_str
  else:
    return date_str

ufo['datetime'] = ufo['datetime'].apply(fix_datetime_format)
ufo['datetime'] = pd.to_datetime(ufo['datetime'], infer_datetime_format= True)
ufo['year'] = pd.DatetimeIndex(ufo['datetime']).year
ufo.sort_values(by = 'datetime', ascending = True, inplace = True)

ufo_final = ufo['year'].value_counts().rename_axis('year').to_frame('ufo_counts')
ufo_final = ufo_final.sort_values(by = 'year').reset_index()
ufo_final = ufo_final.loc[30:][:].reset_index()
ufo_final = ufo_final.drop('index', axis = 1)
```

The TMBD dataset contained lots of information so the first step was to winnow it down to the interesting headers. Unlike the UFO dataset, this one contained duplicate entries and lots of missing data. Using the Pandas drop function, I removed all entries in which the genre or the revenue was missing. Then to create a subset of only science-fiction movies, I created a function that outputs a new DataFrame of only movies in a particular genre. The last step was to sum the revenues (adjusted for inflation) for all the movies in a particular year, which I created another function for. 

```markdown
movies = pd.read_csv('tmdb_movies_data.csv')
movies.head()
```

![image](https://user-images.githubusercontent.com/92558174/141885163-3d298612-b0df-46d8-be64-b430eb610239.png)

```
movies.drop(['id', 'cast', 'homepage', 'director', 'tagline', 'keywords', 'overview', 'runtime',
'production_companies', 'vote_count', 'vote_average'], axis = 1, inplace = True)

filtered_movies = movies.drop_duplicates()
genre_drop = filtered_movies[filtered_movies['genres'].isnull() == True]
filtered_movies = filtered_movies.drop(genre_drop.index)
rev_drop = filtered_movies[filtered_movies['revenue'] == 0]
filtered_movies = filtered_movies.drop(rev_drop.index)
filtered_movies = filtered_movies.reset_index()
filtered_movies = filtered_movies.drop('index', axis=1)

def genre_movies(genre):
  temp = filtered_movies
  for i in range(len(filtered_movies)):
    if genre not in temp.loc[i]['genres']:
      temp = temp.drop(index=i)
  temp = temp.sort_values(by = 'release_year')
  return temp

scifi_movies = genre_movies('Science Fiction')

def genre_rev(genre_movies):
  unique_years = genre_movies['release_year'].unique()
  temp = pd.DataFrame(columns = ['year', 'adj_rev_total'])
  temp['year'] = unique_years
  r_vals = []
  for i in range(len(unique_years)):
    r = scifi_movies.loc[scifi_movies['release_year'] == unique_years[i], 'revenue_adj'].sum()
    r_vals.append(r)
  temp['adj_rev_total'] = r_vals
  return temp

scifi_rev = genre_rev(scifi_movies)
```

The US Census dataset was the easiest to work with. The prepration consisted only of converting the dates to datetime objects to extract the year and converting the population numbers from strings to float values.

```markdown
pop = pd.read_excel('us population.xlsx')

pop['Date'] = pd.to_datetime(pop['Date'])
pop['year'] = pd.DatetimeIndex(pop['Date']).year

mil_pop = []
for i in range(len(pop)):
  mil_pop.append(float(pop.loc[i]['Population'][:-8]))
pop['us_pop_mil'] = mil_pop

pop.drop(['Date', 'Population'], axis = 1, inplace = True)
```


The last step was to create a final DataFrame with all of the variables necessary for the regression models. For this, I used the Pandas merge() function twice, joining on the shared year header.

```markdown
df_final = pd.merge(ufo_final ,scifi_rev, how = 'left')
df_final = pd.merge(df_final, pop, how = 'left')
df_final = df_final.fillna(0)
df_final['adj_rev_mil'] = df_final['adj_rev_total'] / 1000000
df_final.head(10)
```

![image](https://user-images.githubusercontent.com/92558174/141859358-838200cc-d7e6-4af7-b4bd-17e6383e9555.png)


### Time Series Graphs

![image](https://user-images.githubusercontent.com/92558174/141721215-3fc2695b-7aba-4f78-8535-fc3b0ae293f9.png)


### Correlation Results
```
from scipy.stats import pearsonr
pearsonr(df_final['ufo_counts'], df_final['adj_rev_mil'])
```
![image](https://user-images.githubusercontent.com/92558174/141887844-1c5a39e8-3f7f-4da8-ae0d-f03f44d8b3a8.png)

With a coefficient of +0.77 and a p-value of essentially 0, we see a fairly strong relationship between the number of UFO sightings and the adjusted revenue of Sci-Fi movies in the US from 1960 to 2013.

### Linear Regression Results
Scatterplot and least-squares regression line with a 95% confidence interval

![image](https://user-images.githubusercontent.com/92558174/141721549-5026871c-b1dd-4924-9e3e-8d59054b547f.png)

Here, we can see that there is indeed a positive correlation. There appears to be a significant horizontal cluster of data points with near-zero sightings and yearly revenues of sub $2 billion. And for years with relatively large revenue sums, the relationship to UFO sightings is even more pronounced. This may be due to a blockbuster-effect of certain films like Stars Wars and E.T. which gained widespread popularity and broke box office records.

#### Univariate OLS
```markdown
model = ols('ufo_counts ~ us_pop_mil', data = df_final).fit()
print(model.summary())
```
![image](https://user-images.githubusercontent.com/92558174/141722252-109dea83-e2b4-41d6-b358-ca640b3e1c3d.png)

The zero p-value confirms the correlation we calculated above. The R-Squared value of 0.587 indicates that movie revenue is not half-bad as a predictor of UFO sightings!


#### Multivariate OLS
```markdown
model2 = ols('ufo_counts ~ us_pop_mil + adj_rev_mil', data = df_final).fit()
print(model2.summary())
```
![image](https://user-images.githubusercontent.com/92558174/141722375-79531167-5fd7-4494-ba06-5249f306d793.png)

Taking into account population, we can see that we still get a p-value of less that 5% for the revenue variable! And an adjusted R-Squared value of 0.736 means that even more of the year to year differences in sightings are accounted for by the two-variable model.



## Concluding Thoughts and Analysis Limitations

From the two-variable regression model, we can see that there is _indeed_ a correlation between the number of reported UFO sightings and the adjusted revenue of science-fiction movies in the US from 1960 to 2013. However, **correlation means association, NOT causation**. We cannot attribute the rise of sightings exclusively to the growing popularity of the film genre or the population changes of the time. This analysis is far too narrow in scope to come away with any definitive causal conclusions. One confounding variable that I suspect affected both sightings and film revenue the explosion of the internet as well as social media in the late 1990s and early 2000s. Other limitations to consider include:

- This analysis is limited to the time period from 1960 to 2013
- Over half of the TMDB dataset remains unused due to missing data
- The science fiction genre encompasses more than just space, aliens and UFOs
- Almost every movie has multiple genres
- Movie revenue is international revenue, not domestic 
- Cinema revenue may not be the ideal metric to track a film's cultural impact and reach
- This analysis does not take into account the effect of movies beyond the year of release


Similar analyses have been done by other data science enthusiasts but this one shows that even after accounting for population numbers, the relationship between UFO sightings and the box office performance of Sci-Fi movies is still statistically significant!

.
