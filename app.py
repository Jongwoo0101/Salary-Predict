import streamlit as st
import pandas as pd
import joblib

# 모델, 인코더, 정확도 로드
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
r2_score = joblib.load("model/r2_score.pkl")

# 앱 타이틀
st.title("💼 AI 직무 연봉 예측기")

# 직무 선택
job = st.selectbox("직무", ['Data Scientist', 'Machine Learning Engineer', 'AI Engineer'])

# 연도 입력
year = st.text_input("예측 연도", value="2025")

# 경력 수준 선택 (설명 포함)
experience_display = {
    'Entry-level / Junior (EN)': 'EN',
    'Mid-level / Intermediate (MI)': 'MI',
    'Senior-level / Expert (SE)': 'SE',
    'Executive-level / Director (EX)': 'EX'
}
experience_choice = st.selectbox("경력 수준을 선택하세요:", list(experience_display.keys()))
experience_level = experience_display[experience_choice]

# 고용 형태 선택 (설명 포함)
employment_display = {
    'Full-time (FT)': 'FT',
    'Part-time (PT)': 'PT',
    'Contract (CT)': 'CT',
    'Freelance (FL)': 'FL'
}
employment_choice = st.selectbox("고용 형태를 선택하세요:", list(employment_display.keys()))
employment_type = employment_display[employment_choice]

# 거주 국가, 회사 위치 입력
employee_residence = st.text_input("거주 국가 (ISO 코드, 예: US)", value='US')
company_location = st.text_input("회사 위치 (ISO 코드, 예: US)", value='US')

# 원격 근무 비율 선택
remote_ratio = st.slider("원격 근무 비율 (%)", 0, 100, 100, step=50)

# 회사 규모 선택 (설명 포함)
company_display = {
    'Small (S, 50명 이하)': 'S',
    'Medium (M, 50~250명)': 'M',
    'Large (L, 250명 이상)': 'L'
}
company_choice = st.selectbox("회사 규모를 선택하세요:", list(company_display.keys()))
company_size = company_display[company_choice]

# 예측 버튼
if st.button("예측하기"):
    if not year.isdigit():
        st.warning("⚠️ 연도를 숫자로 입력해주세요.")
    else:
        # 입력 데이터프레임 생성
        input_df = pd.DataFrame({
            'work_year': [int(year)],
            'experience_level': [experience_level],
            'employment_type': [employment_type],
            'job_title': [job],
            'employee_residence': [employee_residence],
            'remote_ratio': [remote_ratio],
            'company_location': [company_location],
            'company_size': [company_size]
        })

        # 예측 실행
        input_encoded = encoder.transform(input_df)
        prediction = model.predict(input_encoded)[0]

        # 결과 출력
        st.success(f"🎯 당신의 {year}년 예상 연봉은 **${int(prediction):,} USD** 입니다.")
        st.info(f"📈 모델 정확도 (R² 기준): **{r2_score:.2%}**")
