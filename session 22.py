import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


pd.options.display.max_columns = 5
pd.options.display.max_rows = 100000
data = pd.read_csv('bank-full.csv')

data = data.drop('duration',axis=1)
data.head()
columns = data.select_dtypes(include=[object]).columns
data=pd.concat([data,pd.get_dummies(data[columns])],axis=1)
#remove all integer and float values
data = data.drop(['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1)
data.head()
#drop strings
data=data.drop(['housing_yes','housing_no','housing_unknown'],axis=1)
def func1(var1):
    if(var1 =='yes'):
        return(1)
    elif(var1=='no'):
        return(0)
    elif(var1=='unknown'):
        return (2)
data['New_housing'] = data['housing'].apply(func1)
data = data.drop('housing',axis=1)

def func2(var1):
    if(var1 =='admin.'):
        return(1)
    elif (var1 == 'blue-collar'):
        return (2)
    elif (var1 == 'entrepreneur'):
        return (3)
    elif (var1 == 'housemaid'):
        return (4)
    elif (var1 == 'management'):
        return (5)
    elif (var1 == 'retired'):
        return (6)
    elif (var1 == 'self-employed'):
        return (7)
    elif (var1 == 'services'):
        return (8)
    elif(var1=='student'):
        return(9)
    elif (var1 == 'technician'):
        return (10)
    elif (var1 == 'unemployed'):
        return (11)
    elif(var1=='unknown'):
        return(12)


data = data.drop(['job_admin.','job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','job_unknown'],axis=1)
data['New_job'] = data['job'].apply(func2)
data = data.drop('job',axis=1)

data=data.drop(['default_yes','default_unknown','default_no'],axis=1)
data['New_default'] = data['default'].apply(func1)
data = data.drop('default',axis=1)
data=data.drop(['loan_no','loan_unknown','loan_yes'],axis=1)
data['New_loan'] = data['loan'].apply(func1)
data = data.drop('loan',axis=1)

def func3(var1):
    if(var1 =='cellular'):
        return(1)
    elif(var1=='telephone'):
        return(0)

data=data.drop(['contact_cellular','contact_telephone'],axis=1)
data['New_contact'] = data['contact'].apply(func3)
data = data.drop('contact',axis=1)

def func4(var1):
    if(var1 =='jan'):
        return(1)
    elif (var1 == 'feb'):
        return (2)
    elif (var1 == 'mar'):
        return (3)
    elif (var1 == 'apr'):
        return (4)
    elif (var1 == 'may'):
        return (5)
    elif (var1 == 'jun'):
        return (6)
    elif (var1 == 'jul'):
        return (7)
    elif (var1 == 'aug'):
        return (8)
    elif(var1=='sep'):
        return(9)
    elif (var1 == 'oct'):
        return (10)
    elif (var1 == 'nov'):
        return (11)
    elif(var1=='dec'):
        return(12)

data = data.drop(['month_apr','month_aug','month_dec','month_jul','month_jun','month_mar','month_may','month_nov','month_oct','month_sep'],axis=1)
data['New_month'] = data['month'].apply(func4)
data = data.drop('month',axis=1)

data = data.drop(['subscribed_no','subscribed_yes'],axis=1)
data['New_subscribed'] = data['subscribed'].apply(func1)
data = data.drop('subscribed',axis=1)

def func5(var1):
    if(var1 =='success'):
        return(1)
    elif(var1=='failure'):
        return(0)
    elif (var1=='nonexistent'):
        return (2)

data = data.drop(['poutcome_failure','poutcome_success','poutcome_nonexistent'],axis=1)
data['New_poutcome'] = data['poutcome'].apply(func5)
data = data.drop('poutcome',axis=1)


def func6(var1):
    if(var1 =='single'):
        return(1)
    elif(var1=='married'):
        return(2)
    elif (var1=='divorced'):
        return (3)
    elif (var1=='unknown'):
        return (0)

data = data.drop(['marital_divorced','marital_married','marital_single','marital_unknown'],axis=1)
data['New_marital'] = data['marital'].apply(func6)
data = data.drop('marital',axis=1)

def func7(var1):
    if(var1 =='mon'):
        return(1)
    elif(var1=='tue'):
        return(2)
    elif (var1=='wed'):
        return (3)
    elif (var1=='thu'):
        return (4)
    elif (var1=='fri'):
        return (5)

data = data.drop(['day_of_week_fri','day_of_week_mon','day_of_week_thu','day_of_week_tue','day_of_week_wed'],axis=1)
data['New_day_of_week'] = data['day_of_week'].apply(func7)
data = data.drop('day_of_week',axis=1)
def func8(var1):
    if(var1 =='basic.4y'):
        return(1)
    elif (var1 == 'basic.6y'):
        return (2)
    elif (var1 == 'basic.9y'):
        return (3)
    elif (var1 == 'high.school'):
        return (4)
    elif (var1 == 'illiterate'):
        return (5)
    elif (var1 == 'professional.course'):
        return (6)
    elif (var1 == 'university.degree'):
        return (7)
    elif (var1 == 'unknown'):
        return (0)

data = data.drop(['education_basic.4y','education_basic.6y','education_basic.9y','education_high.school','education_illiterate','education_professional.course','education_university.degree','education_unknown'],axis=1)
data['New_education'] = data['education'].apply(func8)
data = data.drop('education',axis=1)

x = data.iloc[:,0].values.reshape(-1, 1)
y = data.iloc[:,8].values.ravel()
y.ravel()
nd = StandardScaler()
nd.fit(x)

x = nd.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
lr = KNeighborsClassifier()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
for i in range(len(y_pred)):
    y_pred[i] == int(y_pred[i])
for j in range(len(y_test)):
    y_test[j] == int(y_test[j])
accuracies = metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
kappaScores = metrics.cohen_kappa_score(y1=y_test,y2=y_pred)
f1scores = metrics.f1_score(y_true=y_test,y_pred=y_pred,average='macro')



print("accuracy:",accuracies,"\n Kappa:",kappaScores,"\n F1:",f1scores)
