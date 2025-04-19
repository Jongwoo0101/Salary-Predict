# TODO: 모델 정확도 높이기 -> 직무가 3가지만 있는게 아니고 더 있음
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# 1. 데이터 불러오기
df = pd.read_csv("./data/salaries.csv")

# 2. 주요 직무만 필터링
target_jobs = ['Data Scientist', 'Machine Learning Engineer', 'AI Engineer']
df = df[df['job_title'].isin(target_jobs)]

# 3. 특성과 타겟 설정
X = df[['work_year', 'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
y = df['salary_in_usd']

# 4. 범주형 특성 인코딩
categorical_features = ['experience_level', 'employment_type', 'job_title',
                        'employee_residence', 'company_location', 'company_size']
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 인코딩 및 모델 학습
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_encoded, y_train)

# 7. 정확도 평가
y_pred = model.predict(X_test_encoded)
r2 = r2_score(y_test, y_pred)

# 8. 모델, 인코더, 정확도 저장
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(r2, "model/r2_score.pkl")

print(f"✅ 모델 학습 및 저장 완료 (정확도 R²: {r2:.4f})")
