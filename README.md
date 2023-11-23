# Market-Basket-Data-Analytics

## INTRODUCTION:
Market Basket Analysis is a data mining technique used quite frequently by the shop-owners and retailers to understand the shopping behaviour and the purchasing pattern of their customers. This project contains data analysis and data visualization which is helpful in understanding the customers' purchase behavior and also analyzes the number of accidents in different city corresponding to the sales of liquor in that state.

In simpler terms, it helps business owners get the insights of the most frequently bought products and combination of these products.

### NOTE:
Market Basket Analysis does not rely on any assumption like linearity, or normality which are often violated during linear-based techniques.

## PROBLEM STATEMENT:
We have to analysis and provide insights into the purchasing behavior of customers and to help retailers make informed decisions about their marketing and sales strategies which will ultimately increase the profitability of the retailers.

## DATASET:
- Dataset used in this analysis contains the monthly sales data for 2019:

## LOGIC BEHIND MARKET BASKET ANALYSIS:
Primary aim of this analysis is to find the relation between the items being purchased by the customer. This technique primarily computes **If-Then** clause i.e,

> ***If[A] then[B] = 'the item on the left(A) is likely to be ordered more often with the item on the right(B)'***

***Example 1:***

If[Bread] then[Butter] => Customer's are more frequent in buying *Butter* when they buy *Bread*.

## MATHEMATICS & FORMULAE's USED:
1. **Antecedent:** Item on the *LEFT*; primary item being bought. *(Example 1. - Bread)*
2. **Consequent:** Item on the *RIGHT*; secondary item being bought following item on the left. *(Example 2. - Butter)*
3. **Support:** Probability of the occurrence of the antecedent. *(Example 1. -  Probability that the customer will buy the 'Bread')*
4. **Confidence:** Probability of the occurrence of the consequent given tht antecedent has occured. *(Example 1. -  Probability that the customer will buy the 'Butter' when 'Bread' was bought)*
5. **Lift:** Ratio of the support of the left-hand side(antecedent, *Bread*) and it's co-occurrence with the right-hand side(consequent, *Butter*), divided by the probability that the left-hand side(antecedent, *Bread*) and right-hand side(consequent, *Butter*) co-occur together when they are independent.
6. **Zhang's Metric:** Zhang metric or F-measure can be used to evaluate the performance of association rule mining algorithms. This metric's value ranges from -1 to 1 to represent both positive and perfect associations. This metric helps in determining the specific items which must not be put together.
7. **Centrality:** This is a crucial concept as it help in determining the most important node in the graph. Importance of any node depends on the defination of the *importance*.
8. **Silhouette Coefficient:** Silhouette Score is pretty helpful in calculating the goodness of the clustering techniques. Its value ranges from -1 to 1.
- 1 means clusters are well apart from each other and clearly distinguished.
- 0 means clusters are indifferent from each other and distance between them is not significant.
- -1 means clusters are assigned incorrectly.

## APRIORI ALGORITHM:
This is the most popular data-mining algorithm used to extract frequent patterns in the datasets. Association rule is used extensively in this algorithm for determining relations among variables in large datasets.
<p align = 'center'>
  <img src = "./Formula's/Apriori Algorithm.png" alt = 'Apriori Algorithm' width = '400' height = '400'>
</p>
See link in the references for detailed understanding.

## DATA VISUALIZATION:

### Customer Segment:
First visualization corresponds to that of the TOP 25 bought items by the customers.
<p align = 'center'>
  <img src = "./Data Visualization/Top 25 Bought Items.png" alt = 'Top 25 Bought Items' width = '400' height = '250'>
</p>

Now a HEATMAP is plotted for the TOP 6 products brought by the customers to observe association relations among them.
<p align = 'center'>
  <img src = "./Data Visualization/Heatmap Data Correlation.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

Furthermore, item's support matrix has been plotted for the TOP products brought by the customers to observe support relations among them.
<p align = 'center'>
  <img src = "./Data Visualization/Item's Support Matrix.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

After all these hustle, combining the plot for price and quantity ordered against product gives a great insight about how does product price reduces the demand for any product.
<p align = 'center'>
  <img src = "./Data Visualization/Quantity Ordered, Products, Price.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

We also discussed about centrality in the previous section; for better understanding we can visualize the network corresonding to these item's and get insights about the maximum purchased item.
- Figure corresponding to all the items being sold to the customer.
<p align = 'center'>
  <img src = "./Data Visualization/All Item Network.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

- Figure corresponding to the maximum being purchased by the customer.
<p align = 'center'>
  <img src = "./Data Visualization/Connection of Maximum Purchased Item.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

### Alcohol Sales:
On analyzing the numbers of accidents v/s cities following plot was obtained.
<p align = 'center'>
  <img src = "./Data Visualization/Accidents V.S City.png" alt = 'Accidents V.S City' width = '400' height = '250'>
</p>

And now we will visualize CITIES & COUNTRIES WITH MAXIMUM ALCOHOL SALES.
<p float = 'left'>
  <img src = "./Data Visualization/Top cities with maximum alcohol sales.png" alt = 'Top cities with maximum alcohol sales' width = '503' height = '300'>
  <img src = "./Data Visualization/Top countries with maximum alcohol sales.png" alt = 'Top countries with maximum alcohol sales' width = '503' height = '300'>
</p>

On further investigating, the sales vs city plot we will get 'San Francisco CA' making the highest sales of alcohol.
<p align = 'center'>
  <img src = "./Data Visualization/Sales vs Cities.png" alt = 'Top cities with maximum alcohol sales' width = '400' height = '250'>
</p>

## REFERENCES:
[1] [Research Paper: How to Increase Sales in Retail with Market Basket Analysis](https://www.academia.edu/download/56086206/Article_1.pdf)  
[2] [Market Basket Analysis in Management Research](https://journals.sagepub.com/doi/abs/10.1177/0149206312466147)  
[3] [Centrality](https://towardsdatascience.com/graph-analytics-introduction-and-concepts-of-centrality-8f5543b55de3)  
[4] [Silhouette Coefficient](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c)  
[5] [Apriori Algorithm](https://www.javatpoint.com/apriori-algorithm)
