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