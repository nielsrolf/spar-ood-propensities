Can you introduce two buttons: (1) 'unexpected spillover' button and a (2) 'symmetry' button. Both buttons can not be pressed at the same time.
(1) All the current 'table below' functions should only appear when one presses the 'unexpected spillover' button. If the button is not pressed the current 'table below' should not be shown, also the 'highlight everything from table' and 'highlight prompted' should only appear if the first button is pressed.
(2) If the second button is pressed, a table should appear. The idea behind the table is this (exact table structure will be given after idea): 
Idea:
If the elicited propensity (propensity i) +/- scores significantly different than the base model on an eval (propensity j), then we check the scores of the elicited propensity j +/- (pole) on the eval for propensity i.
If pole i+ scores higher than the base on propensity j's eval, there seems to be a positive correlation: thus, we also expect 
(a) pole i- to score lower (than the base) on propensity j's eval,
(b) pole j+ to score higher (than the base) on propensity i's eval, and
(c) pole j- to score lower (than the base) on propensity i's eval.
If pole i+ scores lower (than the) base on propensity j's eval, we expect the same.
If pole i+ scores lower than the base on propensity j's eval, there seems to be a negative correlation; thus we expect (a), (b) and (c) with flipped signs.
Notes:
All of this obviously depends on the metric / definition of 'X is significantly higher than Y'.
I'd start by trying absolute difference X-Y > threshold and relative difference X/Y > threshold for the start.
The user of the html page should have one button to switch between absolute and relative difference - and two variables: one for each threshold.
Table structure: 
For each 'cluster' (poles i +/- with eval j together with poles j +/- with eval i : this is denoted as the i,j-cluster, and if only + or - is currently shown, then only that is taken into account) the table should contain the absolute or relative difference between the numbers (if + and - are given for i and j, this means you show 4 numbers per cluster) the number/cell in the table should be colored according to the suggested correlation (i.e., the difference: if absolute difference > 0 or relative difference > 1: green, if absolute difference < 0 or relative difference < 1: red - use a colormap for this). when hovering over a line in the table the corresponding matrix cells should be highlighted pink and vice versa, if a matrix cell is hovered over, all matrix cells in this cluster should be highlighted pink and the corresponding table line should rise to the top.


