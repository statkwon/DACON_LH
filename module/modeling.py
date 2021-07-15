from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import export_graphviz  
import pydotplus
from IPython.display import Image
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

def modeling(df, model_ls=['RF','GBC','DT']):
    
    df = df.groupby('단지코드').first().reset_index() # drop_duplicates() 같은 역할
    X = df.drop(['등록차량수'],axis=1)
    y = df['등록차량수']
    X = X.loc[:,X.dtypes!='object']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(random_state=0)
    gbrt = GradientBoostingRegressor(random_state=0)
    dt = DecisionTreeRegressor(random_state=0)
    message = 'You have to choose model type'
    model = {model_ls=='RF':rf, model_ls=='GBC':gbrt, model_ls=='DT':dt}.get(True, message)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return(mae(y_test, y_pred))

def submit(train,test):
    train = train.groupby('단지코드').first().reset_index()
    test = test.groupby('단지코드').first().reset_index()
    y_train = train['등록차량수']
    X_train = train.drop(['등록차량수'],axis=1)
    X_train = X_train.loc[:,X_train.dtypes!='object']
    X_test = test.loc[:,test.dtypes!='object']
    code = test['단지코드']
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    result = pd.concat([pd.DataFrame(code).rename(columns={'단지코드':'code'}), pd.DataFrame(y_pred, columns=['num'])], axis=1)
    result = pd.concat([result, pd.DataFrame([['C2675',0],['C2335',0],['C1327',0]], columns=['code','num'])], axis=0).reset_index(drop=True)
    return(result)
