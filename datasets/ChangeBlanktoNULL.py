import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv("hair_salon_no_show_wrangled_df.csv")

# 모든 공백 또는 결측치를 NULL(np.nan)로 처리
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# 저장: NULL 값을 SQL이 인식할 수 있도록 'NULL' 문자열로 저장
df.to_csv("hair_salon_no_show_wrangled_df_cleaned.csv", index=False, na_rep='NULL')

print("✅ 완료: 공백을 NULL로 변환하고 cleaned CSV 파일 저장됨.")
