import pandas as pd

'''
마지막 예약 정보(예약 요일, 예약 스태프, 예약 카테고리, 예약 시간대) 및 현재 예약 정보 결측치 처리

wrangled_data = pd.read_csv('datasets/hair_salon_no_show_wrangled_df.csv')

# 1. is_first_visit: 과거 예약 정보가 모두 결측이면 첫 방문자
wrangled_data['is_first_visit'] = (
    wrangled_data[['last_staff', 'last_category', 'last_tod', 'last_dow']]
    .isnull()
    .all(axis=1)
    .astype(int)
)

# 2. 과거 이력 결측치 → 'Unknown' 처리
wrangled_data['last_staff'].fillna('Unknown', inplace=True)
wrangled_data['last_category'].fillna('Unknown', inplace=True)
wrangled_data['last_tod'].fillna('Unknown', inplace=True)
wrangled_data['last_dow'].fillna('Unknown', inplace=True)

# 3. 현재 예약 정보(book_tod) 결측치 → 최빈값으로 채우기
most_common_tod = wrangled_data['book_tod'].mode()[0]
wrangled_data['book_tod'].fillna(most_common_tod, inplace=True)

# 4. 결측치 업데이트 저장
wrangled_data.to_csv('preprocessing/hair_salon_data.csv', index=False)
'''

# 히트맵 생성
hair_salon_data = pd.read_csv('dataset/processed/hair_salon_data.csv')

print(hair_salon_data.head())
print(hair_salon_data.columns)
