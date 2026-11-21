import numpy as np
import pandas as pd

# 1. 파일 열기
d = np.load("../data/splits/UCI_AirQuality_windows.npz")
x_train = d["X_train"]
x_test = d["X_test"]
names = list(d["sensor_names"])
mean, std = d["mean"], d["std"]

# 2. 전체 모양 확인
print("모양 (창 개수, 시간, 센서):", x_train.shape)
print("테스트 모양", x_test.shape)
print("값 범위:", x_train.min(), "~", x_train.max())   # ±4 안에 있어야 정상

# 3. 첫 번째 창(24시간)을 표로 보기 - 정규화된 값
first = pd.DataFrame(x_train[0], columns=names)
print("\n[첫 번째 창 - 정규화된 값]")
print(first.round(2))

# 4. 같은 창을 실제 단위로 되돌려 보기
first_real = pd.DataFrame(x_train[0] * std + mean, columns=names)
print("\n[첫 번째 창 - 실제 단위]")
print(first_real.round(2))

# 5. 센서별 평균/표준편차 (정규화가 잘 됐으면 평균≈0, 표준편차≈1)
print("\n[센서별 통계 - 정규화된 값]")
print(pd.DataFrame({
    "mean": x_train.mean(axis=(0, 1)),
    "std": x_train.std(axis=(0, 1)),
}, index=names).round(3))