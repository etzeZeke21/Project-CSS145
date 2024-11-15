# Project-CSS145

# Game Reviews Analysis
 by: 
### John Kenneth Alon
### 	Rob Eugene Dequiñon
### Neil Andrew Mediavillo
### Emmanuel Villanosa
### Ezekiel Martin

<hr>
   
The dataset provides reviews from the Steam website across 11 games namely: Arma 3, Counter Strike, Counter Strike Global Offensive, Dota 2, Football Manager 2015, Garry’s Mod, Grand Theft Auto V, Sid Meiers Civilization 5, Team Fortress 2, The Elder Scrolls V, and Warframe. Each review comes alongside numerous pieces of data, including the number of people who marked the review helpful, the number of people who marked the review funny, the number of friends the reviewer has on the site, etc. One of the more important pieces of data, however, is the number of hours that the reviewer played the game that they are reviewing.

## The project succesfully created the following:
  A sentimental analysis model that will analyze a review and decide if the review is reocmmmended or not recommended, using the review text. <br>
  A helpful review classification that can predict the helpfulness score, using the number of voted helpfulness, number of voted funny, total game hours, achievement progress, and number of comments.<br>
  A time series analysis that uses the date posted and total game hours to track how the sentiment of reviews or helpfulness score changes over time.<br>

## Conclusion:
  The sentimental analysis model showed great results in predicting the review text if the review was recommended or not. By adding a few more data it could be used to differentiate which reviews are useful or helpful for other players that may seek information about the game.<br>

  The helpful review classification model used the data groups: number of voted helpfulness, number of voted funny, total game hours, and number of comments to try and predict the helpfulness percentage of the review. According to the random forest classification the only data group that had significant impact was number of voted helpfulness, and the other data groups had little to no importance in the helpfulness percentage. Though the classification model still had a high accuracy percentage.<br>

  The time analysis model analyzed the trends between the games using the dates when the review was posted and the total game hours of the reviewers.The trends were all decreasing which also points to the fact that these games were released at around 10 years ago so after their initial spike and popularity, the reviews are slowing down. But it is still suprising that even after 5 years they are still getting reviews. The model can be used to try and predict the trends until now and it would show that the reviews are on a steady decline from its initial release.<br>
