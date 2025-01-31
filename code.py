import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title("阻尼振盪分析")  # 應用程式標題

# 上傳文件
uploaded_file = st.file_uploader("上傳您的 CSV 檔案", type=["csv"])  # 上傳檔案

if uploaded_file is not None:
    # 如果文件已上傳，嘗試讀取數據
    try:
        data = pd.read_csv(uploaded_file)  # 讀取 CSV
        st.write("### 上傳的資料")  # 顯示上傳的資料標題
        st.dataframe(data)  # 顯示資料表

        # 確保文件中包含所需的列
        if 'time' in data.columns and 'displacement' in data.columns:
            # 提取欄位
            time = data['time']
            amplitude = data['displacement']

            # 峰值檢測
            peaks, _ = find_peaks(amplitude)
            peak_times = time[peaks].values.reshape(-1, 1)  # 峰值時間
            peak_amplitudes = amplitude[peaks]  # 峰值振幅
            log_peak_amplitudes = np.log(peak_amplitudes)  # 計算峰值振幅的自然對數

            # 建立峰值資料的 DataFrame
            peak_data = pd.DataFrame({
                '峰值時間': peak_times.flatten(),
                '峰值振幅': peak_amplitudes,
                'ln(峰值振幅)': log_peak_amplitudes
            })
            st.write("### Peak Data with Natural Logarithm")  # 顯示表格標題
            st.dataframe(peak_data)  # 顯示資料表

            # 繪製 振幅 vs 時間 圖
            st.write("### Amplitude vs Time")  # 圖表標題
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(time, amplitude, label='Damped Oscillation')  # 繪製阻尼振盪曲線
            ax1.plot(peak_times.flatten(), peak_amplitudes, 'ro', label='Detected Peaks')  # 繪製峰值
            ax1.set_title('Amplitude vs Time')  # 圖表總標題
            ax1.set_xlabel('Time (s)')  # x 軸標題
            ax1.set_ylabel('Amplitude (m)')  # y 軸標題
            ax1.legend()  # 顯示圖例
            ax1.grid()  # 顯示格線
            st.pyplot(fig1)  # 顯示圖表

            # 線性回歸
            model = LinearRegression()
            model.fit(peak_times, log_peak_amplitudes)  # 訓練回歸模型
            slope = model.coef_[0]  # 斜率
            intercept = model.intercept_  # 截距
            log_amplitude_pred = model.predict(peak_times)  # 預測值
            r_squared = model.score(peak_times, log_peak_amplitudes)  # R 平方值

            # 繪製 ln(峰值振幅) vs 時間 圖
            st.write("### Natural Logarithm of Peak Amplitude vs Time")  # 圖表標題
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.scatter(peak_times, log_peak_amplitudes, color='red', label='Data (Natural Logarithm)')  # 資料點
            ax2.plot(peak_times, log_amplitude_pred, color='blue', label=f'Fit: ln(A) = {slope:.4f}t + {intercept:.4f}')  # 回歸線
            ax2.set_title('Natural Logarithm of Peak Amplitude vs Time')  # 圖表總標題
            ax2.set_xlabel('Time (s)')  # x 軸標題
            ax2.set_ylabel('ln(Peak Amplitude)')  # y 軸標題
            ax2.legend()  # 顯示圖例
            ax2.grid()  # 顯示格線

            # 顯示 R 平方和回歸方程
            text_x = peak_times.mean()
            text_y = log_amplitude_pred.mean()
            ax2.text(text_x + 3, text_y, f'ln(A) = {slope:.5f}t + {intercept:.5f}', fontsize=12)  # 顯示方程
            ax2.text(text_x + 3, text_y - 0.1, f'$R^2$ = {r_squared:.4f}', fontsize=12)  # 顯示 R 平方值
            st.pyplot(fig2)  # 顯示圖表

            # 顯示回歸結果
            st.write("### 回歸結果")  # 回歸結果標題
            st.write(f"斜率 (衰減率): {slope:.5f}")  # 顯示斜率
            st.write(f"截距: {intercept:.5f}")  # 顯示截距
            st.write(f"R 平方值: {r_squared:.4f}")  # 顯示 R 平方值
        else:
            st.error("CSV 文件缺少必要的列（'time' 和 'displacement'）。請檢查文件格式。")
    except Exception as e:
        st.error(f"讀取 CSV 時發生錯誤: {e}")
else:
    st.info("請上傳一個 CSV 檔案。")
