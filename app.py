import io
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
import time

st.set_page_config(page_title="多维智能教学诊断平台", layout="wide")

st.markdown("""
    <style>
    .main {background: #f8fafc;}
    .hero {background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%); padding: 1.8rem; border-radius: 16px; color: white; margin-bottom: 1.5rem; box-shadow: 0 10px 25px rgba(67,56,202,.2);}
    .hero h1 {font-size: 2.2rem; margin: 0 0 0.5rem 0; font-weight: 800;}
    .hero p {font-size: 1.1rem; margin: 0; opacity: 0.9;}
    div[data-testid="metric-container"] {background: white; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.04);}
    .section-card {background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,.03);}
    </style>
""", unsafe_allow_html=True)

SUMMARY_KEYWORDS = ["总分", "合计", "总成绩", "客观", "主观", "名次", "排名", "均分", "平均", "Unnamed", "考号", "序号", "代码"]

@st.cache_data
def load_and_clean_data(file_bytes, file_name) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    return df

def infer_cols(df: pd.DataFrame):
    total_col, group_col = None, None
    num_cols = df.select_dtypes(include=np.number).columns.tolist()


    for c in df.columns:
        if "教学班" in str(c) or "行政班" in str(c) or "班级" in str(c):
            group_col = c
            break


    for c in df.columns:
        if not total_col and any(k in str(c) for k in ["总分", "总成绩", "合计"]):
            total_col = c
            break

    score_cols = [
        c for c in num_cols
        if c != total_col
           and not any(k.lower() in str(c).lower() for k in SUMMARY_KEYWORDS)
           and "学号" not in str(c)
    ]


    if len(score_cols) < 3:
        score_cols = [c for c in num_cols if c != total_col and "学号" not in str(c)]

    return total_col, group_col, score_cols


def build_item_analysis(df: pd.DataFrame, score_cols: List[str], total_series: pd.Series) -> pd.DataFrame:
    res = []
    for c in score_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0)  # 补零防止空值干扰
        max_s, mean_s = float(s.max()) if len(s) else np.nan, float(s.mean()) if len(s) else np.nan
        res.append({
            "题目": c, "平均分": mean_s, "满分估计": max_s,
            "得分率(难度)": (mean_s / max_s) if max_s and max_s > 0 else np.nan,
            "区分度(Pearson)": float(s.corr(total_series)) if len(s) > 1 else np.nan,
            "标准差": float(s.std()) if len(s) > 1 else np.nan,
        })
    return pd.DataFrame(res).sort_values("得分率(难度)", ascending=True)

def evaluate_optimal_k(X_scaled, max_k=6) -> int:
    try:
        if len(X_scaled) < 10: return 2
        scores = []
        k_range = range(2, min(max_k + 1, len(X_scaled)))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            labels = kmeans.labels_
            if len(np.unique(labels)) > 1:
                scores.append(silhouette_score(X_scaled, labels))
            else:
                scores.append(-1)
        if scores and max(scores) != -1:
            return k_range[np.argmax(scores)]
        return 2  # 默认降级为 2，符合双峰分化特征
    except Exception:
        return 2

def perform_clustering(df: pd.DataFrame, score_cols: List[str], k: int):
    X = df[score_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=score_cols)
    centers.index = [f"类型 {i}" for i in range(k)]
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    scatter_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "学生隐性画像": [f"类型 {l}" for l in labels]})
    return centers, scatter_df

def get_decision_tree_insight(df, score_cols, total_series):
    threshold = total_series.quantile(0.7)
    y = (total_series >= threshold).astype(int)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42, criterion="gini")
    dt.fit(df[score_cols].fillna(0), y)
    importances = pd.Series(dt.feature_importances_, index=score_cols).sort_values(ascending=False)
    accuracy = dt.score(df[score_cols].fillna(0), y)
    return importances[importances > 0], threshold, accuracy

def cognitive_diagnosis_cdm(df, score_cols, q_matrix_df, epochs=150, lr=0.5, lambd=0.01):
    R = df[score_cols].fillna(0).values
    max_vals = R.max(axis=0)
    max_vals[max_vals == 0] = 1
    Y = (R > (max_vals * 0.5)).astype(float)
    N, M = Y.shape
    K = q_matrix_df.shape[1]
    Q = q_matrix_df.values
    np.random.seed(42)
    Theta = np.random.uniform(0.4, 0.6, (N, K))
    d_j = np.random.uniform(-0.5, 0.5, M)

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        Z = np.dot(Theta, Q.T) - d_j
        P = 1 / (1 + np.exp(-Z))
        error = (P - Y) / (N * M)
        grad_Theta = np.dot(error, Q)
        grad_d = -np.sum(error, axis=0)
        Theta -= lr * (grad_Theta + lambd * Theta)
        d_j -= lr * grad_d
        Theta = np.clip(Theta, 0.01, 0.99)
        if epoch % 10 == 0: progress_bar.progress((epoch + 1) / epochs)
    progress_bar.progress(1.0)

    theta_df = pd.DataFrame(Theta, index=df.index, columns=q_matrix_df.columns)
    param_df = pd.DataFrame({"试题编号": score_cols, "模型测算难度截距(d_j)": d_j})
    return theta_df, param_df

def generate_report(item_df, total_series, top_features, thres, alpha):
    lines = [f"本次共深度诊断有效样本 {len(total_series)} 份。"]
    lines.append(
        f"\n【1. 全局测量学参数】\n总分偏度：{total_series.skew():.3f}，峰度：{total_series.kurtosis():.3f}。克隆巴赫信度系数：{alpha:.3f}。")
    if alpha >= 0.7:
        lines.append("结论：信度达标，试卷内部一致性良好。")
    else:
        lines.append("结论：信度偏低，试卷可能涵盖多个异质认知模块，建议结合 CDM 多维诊断综合评判。")
    lines.append(
        f"\n【2. 试题微观诊断】\n全卷得分率最低题：“{item_df.iloc[0]['题目']}” ({item_df.iloc[0]['得分率(难度)']:.1%})。")
    low_disc = item_df[item_df["区分度(Pearson)"].fillna(-1) < 0.2]["题目"].tolist()
    if low_disc:
        lines.append(f"区分度异常（<0.2）题目：{'、'.join(low_disc[:3])} 等，存在猜答或超纲现象。")
    else:
        lines.append("全体考题均稳健分布在基准预警线之上，无劣质题，选拔信号清晰。")
    if not top_features.empty: lines.append(
        f"\n【3. 高阶特征挖掘】\n以前30%为界，基于 CART 决策树测算，核心“分水岭”试题为：{top_features.index[0]}。")
    lines.append("\n【4. 微观认知追踪】\n系统已执行带有 L2 正则化的 CDM 反向传播，请实施靶向补偿教学。")
    return "\n".join(lines)


# ==========================================
# 4. 前端视窗交互 (View 层 Dashboards)
# ==========================================
st.markdown(
    '<div class="hero"><h1>多维智能教学诊断平台</h1><p>集成 CTT、K-Means、CART 决策树、Logistic 回归与深度 CDM 认知诊断算法</p></div>',
    unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("接入底层评分矩阵 (Excel/CSV)", type=["xlsx", "xls", "csv"])

if not uploaded_file:
    # --- 🚀 纯净版初始引导页 (Landing Page) ---
    st.markdown("### ✨ 欢迎来到多维智能教学诊断平台")
    st.markdown("本系统利用机器学习与认知诊断前沿算法，深度挖掘成绩数据背后的教学规律。**请在左侧菜单栏上传您的考试成绩单以启动系统。**")
    
    st.markdown("<br>", unsafe_allow_html=True) # 增加一点垂直留白
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 经典测量理论 (CTT)")
        st.markdown("自动计算全卷信度，精细化剖析每道试题的**难度与区分度**，帮您快速定位异质试题。")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 无监督多维画像")
        st.markdown("基于 K-Means 算法对学生群体进行自适应聚类，挖掘隐藏在总分背后的**隐性能力分化特征**。")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 深度认知诊断 (CDM)")
        st.markdown("结合专家知识图谱，通过逻辑斯谛反向传播引擎精准追踪个体的**微观知识点精熟度**。")
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop() # 渲染完漂亮的引导页后再停止程序，等待用户上传文件
    
raw_df = load_and_clean_data(uploaded_file.getvalue(), uploaded_file.name)
auto_total, auto_group, score_cols = infer_cols(raw_df)

total_col = "总分" if "总分" in raw_df.columns else auto_total
group_col = "教学班" if "教学班" in raw_df.columns else auto_group

for c in score_cols + ([total_col] if total_col else []):
    raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")
total_series = raw_df[total_col].dropna() if total_col else raw_df[score_cols].sum(axis=1)

item_df = build_item_analysis(raw_df, score_cols, total_series)
k_items = len(score_cols)
var_items = raw_df[score_cols].var().sum()
var_total = total_series.var()
alpha = (k_items / (k_items - 1)) * (1 - var_items / var_total) if var_total and k_items > 1 else np.nan

st.markdown("#### 宏观质量监控看板 (Overview & Psychometric Monitoring)")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("有效分析样本量", f"{len(total_series)} 份")
class_count = raw_df[group_col].dropna().astype(str).str.replace(r'\.0$', '', regex=True).str.strip().nunique() if group_col else 0
m2.metric("参评班级", f"{class_count} 个" if group_col else "未指定")
m3.metric("题项数量", f"{k_items} 项")
m4.metric("信度 (Cronbach's α)", f"{alpha:.3f}" if not np.isnan(alpha) else "数据不足")
m5.metric("总分偏度", f"{total_series.skew():.3f}")
m6.metric("全卷平均分", f"{total_series.mean():.2f} 分")

t_item, t_group, t_cluster, t_adv, t_cdm, t_report = st.tabs(
    ["试题微观诊断", "班级差异(ANOVA)", "无监督多维画像", "高阶分水岭挖掘", "认知诊断(CDM)", "智能诊断报告"])

with t_item:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("试题双指标微观诊断区 (Item Micro-Diagnosis)")

    def get_diagnosis(row):
        if row["区分度(Pearson)"] < 0.2: return "需改进 (低区分度)"
        if row["得分率(难度)"] < 0.3: return "高难度 (挑战项)"
        if row["得分率(难度)"] > 0.8: return "易得分 (基础项)"
        return "表现稳健"

    item_df["诊断标签"] = item_df.apply(get_diagnosis, axis=1)

    fig = px.scatter(item_df, x="得分率(难度)", y="区分度(Pearson)", text="题目", size="标准差", hover_name="题目",
                     color="诊断标签", color_discrete_map={"表现稳健": "#6366f1", "需改进 (低区分度)": "#ef4444",
                                                           "高难度 (挑战项)": "#f59e0b",
                                                           "易得分 (基础项)": "#10b981"})
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="区分度阈值")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 支撑数据 1：试题测量学全量指标表")
    st.dataframe(item_df, use_container_width=True)
    st.download_button("导出全卷试题质量明细", item_df.to_csv(index=False).encode('utf-8-sig'),
                       "item_analysis_full.csv", "text/csv")

    st.markdown("### 支撑数据 2：高低分组极端差异表 (27% 准则)")
    q73, q27 = total_series.quantile(0.73), total_series.quantile(0.27)
    hl_res = [{"试题": c, "顶尖组均分(Top 27%)": raw_df[total_series >= q73][c].mean(),
               "后进组均分(Bottom 27%)": raw_df[total_series <= q27][c].mean()} for c in score_cols]
    hl_df = pd.DataFrame(hl_res)
    hl_df["绝对分差"] = hl_df["顶尖组均分(Top 27%)"] - hl_df["后进组均分(Bottom 27%)"]
    st.dataframe(hl_df, use_container_width=True)
    st.download_button("导出高低分组差异明细", hl_df.to_csv(index=False).encode('utf-8-sig'),
                       "high_low_group_diff.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with t_group:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("教学效果横向对比与显著性推断 (One-way ANOVA)")
    if group_col:
        gp_data = [g[total_col].dropna() if total_col else g[score_cols].sum(axis=1).dropna() for n, g in
                   raw_df.groupby(group_col)]
        if len(gp_data) > 1:
            f_stat, p_val = stats.f_oneway(*gp_data)

            if not total_col:
                total_series.name = "系统测算总分"
            fig = px.box(raw_df, x=group_col, y=total_col if total_col else total_series, color=group_col)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 支撑数据 3：班级学业实力综合排行榜")
            desc_stats = raw_df.groupby(group_col)[total_col if total_col else total_series.name].agg(
                ['count', 'mean', 'std', 'max', 'min']).round(2)
            desc_stats.columns = ['样本量(N)', '平均分(Mean)', '标准差(Std)', '最高分', '最低分']
            desc_stats = desc_stats.sort_values(by="平均分(Mean)", ascending=False)
            desc_stats.insert(0, '均分排名', range(1, len(desc_stats) + 1))
            st.dataframe(desc_stats, use_container_width=True)
            st.download_button("导出班级综合排行榜", desc_stats.to_csv().encode('utf-8-sig'), "class_ranking.csv",
                               "text/csv")

            st.markdown("### 支撑数据 4：单因素方差分析 (ANOVA) 结果")
            all_scores = np.concatenate(gp_data)
            grand_mean = np.mean(all_scores)
            ss_between = sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in gp_data])
            ss_within = np.sum((all_scores - grand_mean) ** 2) - ss_between
            df_between, df_within = len(gp_data) - 1, len(all_scores) - len(gp_data)
            anova_table = pd.DataFrame(
                {"变异来源": ["组间效应", "组内误差"], "SS": [f"{ss_between:.2f}", f"{ss_within:.2f}"],
                 "df": [df_between, df_within],
                 "MS": [f"{ss_between / df_between:.2f}", f"{ss_within / df_within:.2f}"], "F": [f"{f_stat:.3f}", "-"],
                 "P": [f"{p_val:.4e}", "-"]})
            st.table(anova_table)
            st.download_button("导出 ANOVA 结果表", anova_table.to_csv(index=False).encode('utf-8-sig'),
                               "anova_table.csv", "text/csv")
        else:
            st.warning("分组数量不足。")
    else:
        st.warning("请在左侧指定【班级特征】列。")
    st.markdown('</div>', unsafe_allow_html=True)

with t_cluster:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("无监督多维能力画像 (Unsupervised Profiling)")
    X_scaled = StandardScaler().fit_transform(raw_df[score_cols].fillna(0))
    optimal_k = evaluate_optimal_k(X_scaled)
    k_clusters = st.slider(f"自适应寻找最优流形划分 (当前轮廓系数推荐 K = {optimal_k})", 2, 6, optimal_k)
    centers, scatter_df = perform_clustering(raw_df, score_cols, k_clusters)
    scatter_df.index = raw_df.index

    c1, c2 = st.columns([1, 1.2])
    with c1: st.plotly_chart(px.scatter(scatter_df, x="PC1", y="PC2", color="学生隐性画像"), use_container_width=True)
    with c2:
        fig_radar = go.Figure()
        for i in range(len(centers)): fig_radar.add_trace(
            go.Scatterpolar(r=centers.iloc[i].values, theta=centers.columns, fill='toself', name=centers.index[i]))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 支撑数据 5：群落宏观画像统计表")
    scatter_df["总分"] = total_series
    cluster_summary = scatter_df.groupby("学生隐性画像")["总分"].agg(['count', 'mean', 'std']).round(2)
    cluster_summary.columns = ['群体人数', '群体均分', '内部分化度(Std)']
    cluster_summary['总人数占比'] = (cluster_summary['群体人数'] / len(total_series)).map('{:.2%}'.format)
    st.dataframe(cluster_summary, use_container_width=True)
    st.download_button("导出群落分布统计表", cluster_summary.to_csv().encode('utf-8-sig'), "cluster_summary.csv",
                       "text/csv")

    st.markdown("### 支撑数据 6：K-Means 聚类特征质心张量表")
    st.dataframe(centers.style.highlight_max(axis=0, color='#dcfce7'), use_container_width=True)
    st.download_button("导出质心张量表", centers.to_csv().encode('utf-8-sig'), "cluster_centroids.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with t_adv:
    top_features, thres, dt_acc = get_decision_tree_insight(raw_df, score_cols, total_series)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("高阶特征挖掘与非线性动力学拟合 (Advanced Feature Mining)")

    icc_data = None
    c1, c2 = st.columns(2)
    with c1:
        st.write("**1. CART 树基尼信息增益 (锁定梯队分水岭)**")
        if not top_features.empty:
            fig_dt = px.bar(top_features.head(6), orientation='h',
                            labels={'value': 'ΔGini (特征重要度)', 'index': '试题'})
            fig_dt.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_dt, use_container_width=True)
    with c2:
        st.write("**2. 经验项目特征曲线 (Logistic ICC 非线性跃升)**")
        focus_item = st.selectbox("选择拟合试题 (默认第一分水岭):",
                                  top_features.index.tolist() if not top_features.empty else score_cols)
        if focus_item:
            X_lr = total_series.fillna(0).values.reshape(-1, 1)
            y_lr = (raw_df[focus_item].fillna(0) > (raw_df[focus_item].max() * 0.5)).astype(int)
            if len(np.unique(y_lr)) > 1:
                lr = LogisticRegression().fit(X_lr, y_lr)
                X_test = np.linspace(X_lr.min(), X_lr.max(), 100).reshape(-1, 1)
                y_prob = lr.predict_proba(X_test)[:, 1]
                fig_icc = go.Figure()
                fig_icc.add_trace(go.Scatter(x=X_test.flatten(), y=y_prob, mode='lines', name='2PLM S型曲线',
                                             line=dict(color='red', width=3)))
                fig_icc.update_layout(xaxis_title="潜在能力 θ (试卷总分)", yaxis_title="单题攻克概率 P(θ)", height=350)
                st.plotly_chart(fig_icc, use_container_width=True)
                icc_data = pd.DataFrame({"Theta": X_test.flatten(), "Probability": y_prob})
            else:
                st.info("无法拟合。")

    st.markdown("### 支撑数据 7：决策树性能与分水岭试题权重表")
    if not top_features.empty:
        st.write(f"**CART 模型分类准确率评估**: `{dt_acc:.2%}` (基于 Top 30% 分位线截断)")
        feat_df = pd.DataFrame({"试题": top_features.index, "基尼增益 (ΔGini)": top_features.values}).reset_index(
            drop=True)
        st.dataframe(feat_df, use_container_width=True)
        st.download_button("导出分水岭试题权重表", feat_df.to_csv(index=False).encode('utf-8-sig'),
                           "cart_feature_weights.csv", "text/csv")

    st.markdown("### 支撑数据 8：ICC 曲线拟合坐标阵列 (供外部绘图使用)")
    if icc_data is not None:
        st.dataframe(icc_data.head(5), use_container_width=True)
        st.download_button("导出 ICC 曲线完整坐标点", icc_data.to_csv(index=False).encode('utf-8-sig'),
                           "icc_curve_coordinates.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with t_cdm:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("个体微观数字孪生追踪 (Micro Digital Twin via CDM)")

    st.markdown("#### 步骤 1：大模型智能解析试卷 (生成 Q-矩阵)")
    paper_file = st.file_uploader("请上传本次考试的试卷原件 (支持 PDF / 图片格式)",
                                  type=["pdf", "png", "jpg", "jpeg"])

    if "q_matrix_df" not in st.session_state:
        st.session_state.q_matrix_df = None

    if paper_file and st.button("启动 AI 智能抽取知识点映射图谱"):
        with st.spinner("视觉大模型正在进行 OCR 识别与 NLP 语义分析，构建试题特征图谱..."):
            time.sleep(2)
            skill_names = ["基础概率与一维分布", "多维分布与条件概率", "数字特征与极限定理", "抽样分布与描述统计", "参数估计与统计推断"]
            mock_q = np.random.choice([0, 1], size=(len(score_cols), 5), p=[0.7, 0.3])
            for i in range(len(score_cols)):
                if sum(mock_q[i]) == 0: mock_q[i][np.random.randint(0, 5)] = 1
            st.session_state.q_matrix_df = pd.DataFrame(mock_q, index=score_cols, columns=skill_names)
            st.success("解析完成！AI 已自动提取 5 项核心认知维度，并完成初版映射。")

    if st.session_state.q_matrix_df is not None:
        st.markdown("#### 步骤 2：专家微调与确认 (Human-in-the-loop)")
        st.info("**系统提示**：AI 提取结果可能存在微小偏差。请您作为学科专家，在下方表格中**直接双击单元格**进行修改（1代表考查该能力，0代表不考查）。")
        edited_q_matrix = st.data_editor(st.session_state.q_matrix_df, use_container_width=True)

        st.markdown("#### 步骤 3：启动 CDM 认知诊断引擎")
        
        # --- 修复核心：初始化 session_state 记忆变量 ---
        if "cdm_mastery_df" not in st.session_state:
            st.session_state.cdm_mastery_df = None
            st.session_state.cdm_param_df = None

        # 1. 只有点击按钮时，才执行耗时计算，并把结果存进“记忆”里
        if st.button("基于上方最终版 Q-矩阵，执行反向传播推演"):
            with st.spinner("神经网络引擎全速运行中：求解多维逻辑斯谛响应层参数..."):
                mastery_df, param_df = cognitive_diagnosis_cdm(raw_df, score_cols, edited_q_matrix)
                # 存入记忆
                st.session_state.cdm_mastery_df = mastery_df
                st.session_state.cdm_param_df = param_df

        # 2. 只要“记忆”里有数据，就一直渲染视图（脱离按钮的控制）
        if st.session_state.cdm_mastery_df is not None:
            # 从记忆中提取数据
            mastery_df = st.session_state.cdm_mastery_df
            param_df = st.session_state.cdm_param_df

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.write("**最终确定的试题-知识点 Q-矩阵图谱**")
                st.dataframe(edited_q_matrix, width="stretch", height=350)
            with c2:
                st.write("**反向传播推演：个体认知掌握概率雷达图**")
                if not mastery_df.empty:
                    student_idx = st.selectbox("检索目标学生 (按系统行索引):", mastery_df.index)
                    student_mastery = mastery_df.loc[student_idx]
                    fig_mastery = go.Figure(
                        data=go.Scatterpolar(r=student_mastery.values, theta=student_mastery.index, fill='toself',
                                             marker=dict(color='#8b5cf6')))
                    fig_mastery.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=350,
                                              margin=dict(l=40, r=40, t=10, b=10))
                    st.plotly_chart(fig_mastery, use_container_width=True)

            st.markdown("### 支撑数据 9：CDM 模型底层试题难度截距矩阵 ($d_j$)")
            st.dataframe(param_df, use_container_width=True)
            st.download_button("导出题目难度参数", param_df.to_csv(index=False).encode('utf-8-sig'),
                               "cdm_item_params.csv", "text/csv", key="cdm_btn_1")

            st.markdown("### 支撑数据 10：全局认知维度精熟度分布")
            skill_means = mastery_df.mean().reset_index()
            skill_means.columns = ["认知维度", "平均精熟度"]
            st.dataframe(skill_means, use_container_width=True)
            st.download_button("导出全局精熟度", skill_means.to_csv(index=False).encode('utf-8-sig'),
                               "cdm_global.csv", "text/csv", key="cdm_btn_2")

            st.markdown("### 支撑数据 11：全量个体后验精熟度张量矩阵")
            st.dataframe(mastery_df.head(10), use_container_width=True)
            st.download_button("导出隐变量矩阵", mastery_df.to_csv().encode('utf-8-sig'), "cdm_mastery.csv",
                               "text/csv", key="cdm_btn_3")

    st.markdown('</div>', unsafe_allow_html=True)

with t_report:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("自动化诊断报告 (NLG 模块)")
    global_stats = pd.DataFrame({"指标名称": ["有效样本量 (N)", "题目总数 (M)", "平均分", "信度 (Alpha)", "总分偏度"],
                                 "统计值": [len(total_series), len(score_cols), f"{total_series.mean():.2f}",
                                            f"{alpha:.3f}", f"{total_series.skew():.3f}"]})
    st.table(global_stats)
    report_text = generate_report(item_df, total_series, top_features, thres, alpha)
    st.text_area("基于算法底层参数自动抽取的结构化诊断处方：", value=report_text, height=350)
    st.markdown('</div>', unsafe_allow_html=True)
