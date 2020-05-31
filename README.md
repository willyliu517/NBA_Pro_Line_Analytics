# NBA Pro Line Predictive Analytics
This repo explores machine learning models used to predict outcomes of NBA games. The format of predictions is laid out to follow those laid out by [Pro Line](https://www.proline.ca/). 
### How to play Pro Line: 
A valid Pro Line ticket consists of 3 to 6 events that the player wishes to wager on. 

The player can wager on either the overall outcome of the match:
  * Home Team wins by 11+ points
  * Home Team wins by 6+ points
  * The game is within 5 points
  * Away team wins by 6+ points
  * Away team wins by 11+ points
  
Or the over/under line on the total amount of points scored by both teams:
  * For example, over/under 250 points for the Hawks vs. Thunder game 

The Pro Line ticket will only payout if the player is able to correctly predict all the outcomes he/she chooses to wager on. Thus the more picks a player makes, the higher the potential payout and also the lower probability of a payout. One strategy suggested here is to make multiple tickets but hedge the games of least uncertainty.

We will use game data from the 2009/2010 season to the 2018/2019 season to build two algorithms:
  * Model to predict the outcome of the game - for sake of simplicity, we can reduce this problem to 3 potential outcomes: 
      * Home Team wins by 6+ points
      * The game is within 5 points
      * Away team wins by 6+ points
  * Model to predict the total number of points scored by both teams









 
