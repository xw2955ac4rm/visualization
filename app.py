import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import chardet

# 应用标题
st.title("交互式数据分析与可视化")

# 文件上传
uploaded_file = st.file_uploader("上传一个 CSV 文件进行分析", type=["csv"])

if uploaded_file:
    # 检测文件编码
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

    # 重置文件指针并读取数据
    uploaded_file.seek(0)
    try:
        data = pd.read_csv(uploaded_file, encoding=encoding)
        st.write("### 数据预览")
        st.dataframe(data)

        # 数据筛选
        st.write("### 数据筛选")
        columns = list(data.columns)
        selected_column = st.selectbox("选择需要分析的列", columns)

        if pd.api.types.is_numeric_dtype(data[selected_column]):
            st.write("### 描述性统计")
            st.write(data[selected_column].describe())

            st.write("### 数据分布")
            fig, ax = plt.subplots()
            ax.hist(data[selected_column], bins=20, alpha=0.7)
            ax.set_title(f"{selected_column} 的直方图")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("频数")
            st.pyplot(fig)
        else:
            st.write("选择的列不是数值型数据，无法生成图表。")
    except UnicodeDecodeError:
        st.error("文件编码错误，请保存为 UTF-8 格式后重新上传。")
else:
    st.write("请上传一个 CSV 文件。")
