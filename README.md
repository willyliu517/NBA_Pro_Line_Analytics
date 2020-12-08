# NBA Pro Line Predictive Analytics
This repo explores machine learning models used to predict outcomes of NBA games. The format of predictions is laid out to follow those laid out by [Pro Line](https://www.proline.ca/).

## Overview
1. [How PROLINE Works](#How-to-play-Pro-Line)
2. [Modeling Strategies](#Modeling-Strategies)
    * [Gradient Boosting Machine Approach](#Gradient-Boosting-Machine-Approach)


### How to play PROLNE
A valid Pro Line ticket consists of 3 to 6 events that the player wishes to wager on.

The player can wager on either the overall outcome of the match:
  * Home Team wins by 11+ points
  * Home Team wins by 6+ points
  * The game is within 5 points
  * Away team wins by 6+ points
  * Away team wins by 11+ points

Or the over/under on the total amount of points scored in the game:
  * For example, over/under 250 points for the Hawks vs. Thunder game

The Pro Line ticket will only payout if the player is able to correctly predict all the outcomes picked. Thus the more picks a player makes, the higher the potential payout and also the lower probability of a payout.

### Modeling Strategies
We will use data from the 2009-2010 season to the 2018-2019 season to predict two key outcomes:
  * The overall outcome of the game classified to 3 categories:
      * Home Team wins by 6+ points
      * The game is within 5 points
      * Away team wins by 6+ points
  * The total number of points scored in the game (by both team)

#### Gradient Boosting Machine Approach
The first method considered is building both multi-class and regression GBMs to predict the outcomes listed above. GBM is a machine learning technique that produces a predictive model in the form an ensemble of weak prediction models (typically decision trees). The model is built in a stage-wise fashion similar to other boosting methods. During each stage of the algorithm, the algorithm will try improve upon the residuals from the previous stage utilizing the decision tree built the current stage (every iteration is trying to correct the errors from its predecessor). For more information on GBMs, see the Wikipedia page [here](https://en.wikipedia.org/wiki/Gradient_boosting).
