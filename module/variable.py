import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def gen_size(df):
# 소형세대, 중형세대, 대형세대 변수 생성
    size = pd.DataFrame(columns=['단지코드','소형세대', '중형세대','대형세대'])
    size['단지코드'] = df.단지코드.unique()
    code = size['단지코드']
    for i in range(len(code)):
        house = df[df['단지코드']==code[i]]
        size.loc[i, '소형세대'] = sum(house.loc[house['전용면적']<40,'전용면적별세대수'])
        size.loc[i, '중형세대'] = sum(house.loc[(house['전용면적']>=40)&(house['전용면적']<80) ,'전용면적별세대수'])
        size.loc[i, '대형세대'] = sum(house.loc[house['전용면적']>=80,'전용면적별세대수'])

    size['소형세대'] = size['소형세대'].astype('int64')
    size['중형세대'] = size['중형세대'].astype('int64')
    size['대형세대'] = size['대형세대'].astype('int64')

    df = pd.merge(df, size, on='단지코드', how='left')
    df['소형세대'] = df['소형세대']/df['총세대수']
    df['중형세대'] = df['중형세대']/df['총세대수']
    df['대형세대'] = df['대형세대']/df['총세대수']
    return(df)

def code_name(df):
    # '단지명' 변수 생성
    myhome = pd.read_csv('myhome.csv').drop_duplicates()

    #myhome 데이터의 필요한 부분만 긁어오기
    myhome = myhome[['rnAdres','hsmpNm','hshldCo','suplyTyNm', '공급면적(전용)','임대보증금','임대료', 'x','y']]
    myhome.columns= ['주소', '단지명', '총세대수', '공급유형', '전용면적','임대보증금','임대료','경도','위도']

    #공급유형 및 지역 변수 설정
    myhome.loc[myhome['공급유형']=='50년임대', '공급유형'] = '공공임대(50년)'
    myhome.loc[myhome['공급유형']=='10년임대', '공급유형'] = '공공임대(10년)'
    myhome.loc[myhome['공급유형']=='5년임대', '공급유형'] = '공공임대(5년)'
    myhome['지역'] = myhome['주소'].str.split(' ').str[0]

    # 총세대수 -> 전용면적별세대수합 생성
    myhome_grouped = myhome[['주소','단지명','총세대수']].drop_duplicates()
    myhome_grouped = myhome_grouped.groupby(['단지명','주소'], as_index=False).sum()[['단지명','주소','총세대수']]
    myhome = pd.merge(myhome, myhome_grouped, on=['단지명','주소'], how='left')
    myhome.drop(['총세대수_x'], axis=1, inplace=True)
    myhome.rename(columns={'총세대수_y':'전용면적별세대수합'}, inplace=True)

    # First Match: 지역, 전용면적별세대수합, 공급유형, 전용면적 겹치는 것 중 유니크 382단지
    df = df.drop(['전용면적별세대수합'],axis=1)
    df = df.rename(columns={'전용면적별세대수합_myhome':'전용면적별세대수합'})
    home_merged = pd.merge(df, myhome, on=['지역', '전용면적별세대수합', '공급유형', '전용면적'])
    home_unq = home_merged.groupby(['단지코드']).nunique()
    code_unq = home_unq.index[np.where(home_unq['단지명']==1)]
    name_unq = home_merged[home_merged['단지코드'].isin(code_unq)]['단지명'].unique()
    home_unq.reset_index(drop=True, inplace=True)

    #Second Match: 지역, 공급유형, 전용면적, 임대보증금, 임대료 겹치는 것 중 전용면적별세대수 차이가 적은것
    home_unk = pd.merge(df, myhome, on=['지역','공급유형','전용면적','임대보증금','임대료'])
    home_unk = home_unk[~home_unk['단지코드'].isin(code_unq)]
    home_unk = home_unk[~home_unk['단지명'].isin(name_unq)]
    home_unk['전용면적별세대수차']=abs(home_unk['전용면적별세대수합_x']-home_unk['전용면적별세대수합_y'])
    home_unk = home_unk.groupby(['단지코드','단지명','주소','위도','경도','전용면적별세대수차']).nunique('전용면적')['전용면적'].reset_index()
    home_unk = home_unk.loc[home_unk.groupby('단지코드').전용면적별세대수차.idxmin()]

    first_match = home_merged[home_merged['단지코드'].isin(code_unq)][['단지명','단지코드','주소','위도','경도']].drop_duplicates()
    second_match = home_unk[['단지명','단지코드','주소','위도','경도']].drop_duplicates()
    match = pd.concat([first_match, second_match],axis=0).reset_index(drop=True)

    # Final Match: 남은 것중에 지역과 공급유형이 동일하고 전용면적별세대수합 차이가 가장 적은것
    for_match = df[~df['단지코드'].isin(match['단지코드'])][['단지코드','지역','공급유형','전용면적별세대수합']].drop_duplicates()
    list_match =[]
    for code in for_match['단지코드'].unique():
        for_dict = for_match[for_match['단지코드']==code]
        rg = list(for_dict['지역'].values)
        sp = list(for_dict['공급유형'].values)
        nh = list(for_dict['전용면적별세대수합'].values)[0]
        myhome_match = myhome[(myhome['지역'].isin(rg))&(myhome['공급유형'].isin(sp))]
        myhome_match = list(myhome_match.loc[abs(myhome_match['전용면적별세대수합']-nh).idxmin()][['단지명','주소','위도','경도']])
        list_match.append([code]+myhome_match)

    final_match = pd.DataFrame(list_match, columns=['단지코드','단지명','주소','위도','경도'])
    match = pd.concat([match, final_match],axis=0).reset_index(drop=True)
    df = pd.merge(df, match, on=['단지코드'])
    df = df.drop(['위도','경도'],axis=1) #위도, 경도 결측값 채우기전까지는 빼고 쓰자.
    return(df)

def major_voting(df, colname):
    # 전용면적별세대수에 따른 major voting
    type = df.groupby(['단지코드', colname], as_index=False).agg({'전용면적별세대수':'sum'}).sort_values('전용면적별세대수', ascending=False).drop_duplicates('단지코드').reset_index(drop=True).drop(['전용면적별세대수'], axis=1)
    type.columns = ['단지코드', colname+'_major']
    df = pd.merge(df, type, how='left', on='단지코드')
    return(df)

def total_member(df):
    age_gender = pd.read_csv('한국토지주택공사_임대주택 단지별 연령대별 성별정보_20210511.csv', encoding='CP949')
    age_gender_info = pd.read_csv('age_gender_info.csv')
    age_gender = age_gender.drop(['단지_일련번호','임대주택유형','공급기관명'], axis=1)
    age_gender = age_gender[age_gender['주택유형']=='아파트'].drop_duplicates()
    age_gender = age_gender.drop(['주택유형'], axis=1)
    age_gender.rename(columns={'도로명주소':'주소', '주택명':'단지명'}, inplace=True)
    age_gender = age_gender[age_gender.주소.notnull()]
    age_gender = age_gender.drop_duplicates().reset_index(drop=True) #총세대수0인거 왜날림?
    age_gender = age_gender.drop(['총세대수'], axis=1)
    age_gender['총입주민수'] = age_gender.set_index(['단지명','주소']).apply(sum,axis=1).values
    age_gender = age_gender.drop(['단지명'], axis=1)
    age_gender.loc[:,age_gender.columns.str.contains('대')] = age_gender.loc[:,age_gender.columns.str.contains('대')].mul(1/age_gender['총입주민수'], axis=0)
    df = pd.merge(df, age_gender, on=['주소'], how='left')
    return(df)

def weigted_fee(df):
    df['임대료'] = round(df['임대료'] * df['전용면적별세대수'] / df['전용면적별세대수합'])
    rental_fee = df.groupby('단지코드', as_index=False).agg({'임대료':'sum'})
    rental_fee = rental_fee.rename(columns={'임대료':'가중평균임대료'})
    df = pd.merge(df, rental_fee, how='left', on='단지코드')
    return(df)