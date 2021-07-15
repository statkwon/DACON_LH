import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def Preprocessing(df, return_train=True):
 
    if(return_train): 
        train_error = ['C2085', 'C1397', 'C2431', 'C1649', 'C1036', 'C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']
        df = df[~df['단지코드'].isin(train_error)].reset_index(drop=True)
        # Data의 '전용면적별세대수합' 변수 생성
        
    else:
        test_error = ['C2675', 'C2335', 'C1327']
        df = df[~df['단지코드'].isin(test_error)].reset_index(drop=True)
    # Data의 '임대건물구분 == 상가' 데이터 제거
    df = df[df['임대건물구분'] != '상가'].reset_index(drop=True)
    df.drop(['임대건물구분'], axis=1, inplace=True)

    # Data의 '전용면적별세대수합' 변수 생성
    noh_by_area = df.groupby('단지코드', as_index=False).agg({'전용면적별세대수':'sum'}).rename(columns={'전용면적별세대수':'전용면적별세대수합'})
    df = pd.merge(left=df, right=noh_by_area, how='left', on='단지코드')
    
    df_myhome = df[df['공급유형']!='공공분양']
    noh_by_area = df_myhome.groupby('단지코드', as_index=False).agg({'전용면적별세대수':'sum'}).rename(columns={'전용면적별세대수':'전용면적별세대수합_myhome'})
    df = pd.merge(left=df, right=noh_by_area, how='left', on='단지코드')

    if(return_train):
        df = df[df['지역'] != '서울특별시'].reset_index(drop=True)
        df = df[~df['자격유형'].isin(['F', 'O', 'B'])].reset_index(drop=True)
        df = df[~df['공급유형'].isin(['공공임대(5년)', '공공분양', '장기전세'])].reset_index(drop=True)

    else:# Test Data의 '단지코드' NA 특정값으로 대체
        df.loc[df['단지코드'] == 'C2411', '자격유형'] = 'A'
        df.loc[df['단지코드'] == 'C2253', '자격유형'] = 'D'
        # Test Data의 '임대보증금', '임대료' NA 특정값으로 대체
        df.loc[df['임대보증금'].isnull(), '임대보증금'] = [5787000.0, 5787000.0, 11574000.0]
        df.loc[df['임대료'].isnull(), '임대료'] = [79980.0, 79980.0, 159960.0]
    
    # Data의 '임대보증금', '임대료'의 '-'를 0으로 변환
    df['임대보증금'] = df['임대보증금'].replace('-', 0).astype('float64')
    df['임대료'] = df['임대료'].replace('-', 0).astype('float64')

    # Data의 '도보 10분거리 내 지하철역 수(환승노선 수 반영)' NA 0으로 대체
    df['도보 10분거리 내 지하철역 수(환승노선 수 반영)'] = df['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].fillna(0)
    return(df)

def dummy(train, test, cat_name):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe.fit(train[[cat_name]])
    train_cat = pd.DataFrame(ohe.transform(train[[cat_name]]), columns=ohe.get_feature_names([cat_name]))
    train = pd.concat([train.drop([cat_name],axis=1), train_cat], axis=1)
    test_cat = pd.DataFrame(ohe.transform(test[[cat_name]]), columns=ohe.get_feature_names([cat_name]))
    test = pd.concat([test.drop([cat_name],axis=1), test_cat], axis=1)
    return train, test

def dummy_train(train, cat_name):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe.fit(train[[cat_name]])
    train_cat = pd.DataFrame(ohe.transform(train[[cat_name]]), columns=ohe.get_feature_names([cat_name]))
    train = pd.concat([train.drop([cat_name],axis=1), train_cat], axis=1)
    return train


