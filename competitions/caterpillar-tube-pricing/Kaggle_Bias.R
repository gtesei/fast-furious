
a1 = c(30.02,29.99,30.11,29.97,30.01,29.99)
a2 = c(29.89,29.93,29.72,29.98,30.02,29.98)

a1 = c(0.2030986,
       0.2079454,
       0.208716,
       0.2095649,
       0.2100022,
       0.2151331,
       0.2198506,
       0.2230427,
       0.225096,
       0.2566083,
       0.2630847,
       0.2825056,
       0.2935212)
a2 = c(0.263481,
       0.261631,
       0.265286,
       0.263044,
       0.263987,
       0.262793,
       0.263134,
       0.261335,
       0.261572,
       0.297154,
       0.321223,
       0.27731,
       0.298575)

a2 = a2[a1<0.24]
a1 = a1[a1<0.24]

diff = a1-a2
t.test(x = diff , mu = 0 )
## null hypothesis: the means of the 2 populations are equal.


Assuming the null hypothesis as the means of cross-validation score and Kaggle Public LB score are equal., it results that difference in means with 95 percent confidence lies in the interval [-0.05392342, -0.02951592]. So, I am observing a bias.

