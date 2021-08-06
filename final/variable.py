import pandas as pd
import numpy as np

### Public Data importing(myhome_xyfill은 크롤링 데이터입니다.)
code_age_gender = pd.read_csv('한국토지주택공사_임대주택 단지별 연령대별 성별정보_20210511.csv', encoding='CP949')
myhome_xyfill = pd.read_csv('myhome_xyfill.csv').drop_duplicates()
age_gender_info = pd.read_csv('age_gender_info.csv')

### Function
def code_name(df):
    #myhome 데이터의 필요한 부분만 긁어오기
    myhome = myhome_xyfill[['rnAdres','hsmpNm','hshldCo','suplyTyNm','공급면적(전용)','임대보증금','임대료', 'x','y','준공일자']]
    myhome.columns= ['주소', '단지명', '총세대수', '공급유형', '전용면적','임대보증금','임대료','경도','위도','준공일자']
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
    home_unk = home_unk.groupby(['단지코드','단지명','주소','위도','경도','준공일자','전용면적별세대수차']).nunique('전용면적')['전용면적'].reset_index()
    home_unk = home_unk.loc[home_unk.groupby('단지코드').전용면적별세대수차.idxmin()]
    
    first_match = home_merged[home_merged['단지코드'].isin(code_unq)][['단지명','단지코드','주소','위도','경도','준공일자']].drop_duplicates()
    second_match = home_unk[['단지명','단지코드','주소','위도','경도','준공일자']].drop_duplicates()
    match = pd.concat([first_match, second_match],axis=0).reset_index(drop=True)

    # Final Match: 남은 것중에 지역과 공급유형이 동일하고 전용면적별세대수합 차이가 가장 적은것
    for_match = df[~df['단지코드'].isin(match['단지코드'])][['단지코드','지역','공급유형','전용면적별세대수합']].drop_duplicates()
    list_match =[]
  
    for code in for_match['단지코드'].unique():
        for_dict = for_match[for_match['단지코드']==code]
        rg = list(for_dict['지역'].values)
        sp = list(for_dict['공급유형'].values)
        if (sp[0]=='공공임대(분납)'):
            sp = ['공공임대(10년)'] # Myhome에는 공공임대(분납)이 없으므로
        nh = list(for_dict['전용면적별세대수합'].values)[0]
        myhome_match = myhome[(myhome['지역'].isin(rg))&(myhome['공급유형'].isin(sp))]
        myhome_match = list(myhome_match.loc[abs(myhome_match['전용면적별세대수합']-nh).idxmin()][['단지명','주소','위도','경도','준공일자']])
        list_match.append([code]+myhome_match)

    final_match = pd.DataFrame(list_match, columns=['단지코드','단지명','주소','위도','경도','준공일자'])
    match = pd.concat([match, final_match],axis=0).reset_index(drop=True)
    df = pd.merge(df, match, on=['단지코드'])
    df = df.fillna('0')
    return(df)

def total_member(df):
    # age_gender에서 주소, 단지명, 총입주민수, 주소_mod만 추출하기
    df_address = df[['주소','총세대수','단지명']].drop_duplicates().reset_index(drop=True)
    age_gender = code_age_gender[code_age_gender['주택유형']=='아파트']
    age_gender = age_gender.iloc[:,3:].drop(['주택유형','총세대수'],axis=1) 
    age_gender.rename(columns={'도로명주소':'주소', '주택명':'단지명'}, inplace=True)
    age_gender = age_gender[age_gender['주소'].notnull()]
    age_gender['단지명'] = age_gender['단지명'].str.replace(' ','')
    age_gender['총입주민수'] = age_gender.set_index(['단지명','주소']).apply(sum,axis=1).values
    age_gender = age_gender.groupby(['주소','단지명']).sum().reset_index()
    age_gender.loc[:,age_gender.columns.str.contains('대')] = age_gender.loc[:,
    age_gender.columns.str.contains('대')].mul(1/age_gender['총입주민수'], axis=0)
    age_gender['주소_mod'] = [''.join(elem[:-1]) for elem in age_gender['주소'].str.split()]

    # df에서 주소, 단지명, 주소_mod만 추출하기
    df_address['단지명_origin'] = df_address['단지명']
    df_address['단지명'] = df_address['단지명'].str.replace(' ','')
    df_address['주소_mod'] = [''.join(elem[:-1]) for elem in df_address['주소'].str.split()]
    age_dist = list(age_gender.columns[age_gender.columns.str.contains('대')])
    df_address[age_dist+['총입주민수']] = None
    for i in range(df_address.shape[0]):
        add = df_address['주소'][i]
        cod = df_address['단지명'][i]
        try: 
            df_address.loc[i,age_dist+['총입주민수']] = age_gender.loc[np.where((age_gender['주소']==add)&(age_gender['단지명']==cod))[0][0],age_dist+['총입주민수']]
        except:
            try: 
                add2 = df_address['주소_mod'][i]
                df_address.loc[i,age_dist+['총입주민수']] = age_gender.loc[np.where((age_gender['주소_mod']==add2)&(age_gender['단지명']==cod))[0][0],age_dist+['총입주민수']]
            except:
                try: 
                    df_address.loc[i,age_dist+['총입주민수']] = age_gender.loc[np.where(age_gender['주소']==add)[0][0],age_dist+['총입주민수']]
                except: pass
    # df_check = df_address[df_address.notnull().all(axis=1)]
    #((df_check['총입주민수'] / df_check['총세대수']).astype(float)).describe().round(1) # df_address[df_address.isnull().any(axis=1)] : 1.8
    df_address['지역'] = [elem[0] for elem in df_address['주소'].str.split()]
    for i in range(df_address.shape[0]):
        if ((df_address['총입주민수'][i]!=df_address['총입주민수'][i])|(df_address['총입주민수'][i]==None)):
            df_address['총입주민수'][i] = df_address['총세대수'][i] * 1.8
            reg = df_address['지역'][i]
            df_address.loc[i,age_dist] = age_gender_info.loc[age_gender_info['지역']==reg,age_dist].values[0]
    df_address = df_address.drop(['단지명','총세대수','주소_mod','주소','지역'],axis=1)
    df_address = df_address.rename(columns={'단지명_origin':'단지명'})
    df_address = df_address.drop_duplicates()
    df = pd.merge(df, df_address, on='단지명', how='left').reset_index(drop=True)
    df['총입주민수']=df['총입주민수'].astype(float)
    return(df)
