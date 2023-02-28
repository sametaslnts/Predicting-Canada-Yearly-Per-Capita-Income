import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

print(reg.predict([[2020]]))

plt.xlabel("year")
plt.ylabel("per capita income (US$)")
plt.scatter(df.year, df['per capita income (US$)'], color="red", marker=".")
plt.plot(df.year, reg.predict(df[["year"]]), color="blue")
plt.show()





#def pred(year):  #ax^2+c
 #   a=reg.coef_
  #  x=year
   # c=reg.intercept_

    #result=((a*(x**2))+c)
    #return result
#print(pred(2020))


#def pred(year):
  #  m=reg.coef_
   # b=reg.intercept_

    #result=(m*year+b)
    #return result

#print(pred(2020))
