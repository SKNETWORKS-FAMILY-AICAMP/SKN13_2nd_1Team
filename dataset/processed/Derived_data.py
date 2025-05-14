# import
import pandas as pd
import numpy as np

# data loading
data = pd.read_csv("C:\SKN13SM\SKN13_2nd_1Team\SKN13_2nd_1Team\dataset\processed\hair_salon_data.csv")

# 파생 변수 생성: 첫 방문 
# last_category와 last_staff 변수를 참조해 첫 방문이면 1, 아니면 0으로 인코딩
data['first_visit'] = np.where(
    (data['last_category'] == 'Unknown') & (data['last_staff'] == 'Unknown'),
    1,
    0
)

# 파생 변수 생성: 한 달 내 재방문
# recency 변수를 참조해 한 달 내 재방문이면 1, 아니면 0으로 인코딩
data['is_revisit_30days'] = np.where(
    data['recency'] <= 30,
    1,
    0
)

# 변수 삭제: is_first_visit
# 위에서 생성한 first_visit과 의미가 동일해 제거
data.drop(columns=['is_first_visit'], inplace=True)

# 내용 저장
data.to_csv('C:\SKN13SM\SKN13_2nd_1Team\SKN13_2nd_1Team\dataset\processed\hair_salon_data.csv', index=False)
