import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils.ui import render_sidebar, render_progress, render_nav

st.set_page_config(page_title="Базовая статистика", page_icon="📐", layout="wide")
render_sidebar(current=1)

st.title("📐 Модуль 1: Базовая статистика")
render_progress(current=1)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.header("1. Распределения")

with st.expander("📖 Теория", expanded=True):
    st.markdown("""
**Распределение** описывает как вероятность распределена по возможным значениям случайной величины.

| Распределение | Тип | Параметры | Когда использовать |
|---------------|-----|-----------|-------------------|
| **Нормальное** | Непрерывное | μ, σ | Рост, ошибки измерений, время доставки |
| **Пуассона** | Дискретное | λ | Количество событий за период (заказы/час) |
| **Биномиальное** | Дискретное | n, p | Число успехов из n испытаний |
| **Экспоненциальное** | Непрерывное | λ | Время между событиями |

**Ключевые характеристики:**
- **Математическое ожидание** $E[X]$ — "центр тяжести" распределения
- **Дисперсия** $\\text{Var}(X) = E[(X - \\mu)^2]$ — мера разброса
- **Стандартное отклонение** $\\sigma = \\sqrt{\\text{Var}(X)}$ — в тех же единицах что X
""")

col_ctrl, col_plot = st.columns([1, 3])

with col_ctrl:
    dist_type = st.selectbox(
        "Распределение",
        ["Нормальное", "Пуассона", "Биномиальное", "Экспоненциальное"],
    )
    n_samples = st.slider("Размер выборки", 100, 5000, 1000, step=100)

    if dist_type == "Нормальное":
        mu  = st.slider("μ (среднее)", -10.0, 10.0, 0.0, step=0.5)
        sig = st.slider("σ (std)", 0.1, 5.0, 1.0, step=0.1)
        rng = np.random.default_rng(42)
        data = rng.normal(mu, sig, n_samples)
        food_ex = f"Время доставки: среднее {mu:.0f} мин, std {sig:.0f} мин"

    elif dist_type == "Пуассона":
        lam = st.slider("λ (интенсивность)", 0.5, 30.0, 8.0, step=0.5)
        rng = np.random.default_rng(42)
        data = rng.poisson(lam, n_samples).astype(float)
        food_ex = f"Заказов в час: в среднем {lam:.0f}"

    elif dist_type == "Биномиальное":
        n_tr = st.slider("n (испытаний)", 1, 50, 20)
        p    = st.slider("p (вероятность успеха)", 0.01, 0.99, 0.3, step=0.01)
        rng  = np.random.default_rng(42)
        data = rng.binomial(n_tr, p, n_samples).astype(float)
        food_ex = f"Из {n_tr} пользователей повторно закажут в среднем {n_tr*p:.1f}"

    else:  # Экспоненциальное
        lam2 = st.slider("λ (интенсивность)", 0.1, 2.0, 0.5, step=0.1)
        rng  = np.random.default_rng(42)
        data = rng.exponential(1 / lam2, n_samples)
        food_ex = f"Время между заказами: среднее {1/lam2:.1f} мин"

    mean_v = np.mean(data)
    std_v  = np.std(data)
    st.metric("Среднее", f"{mean_v:.3f}")
    st.metric("Std", f"{std_v:.3f}")

with col_plot:
    fig = px.histogram(
        x=data, nbins=50, marginal="box",
        labels={"x": "Значение", "count": "Частота"},
        title=f"{dist_type} — n={n_samples}",
        color_discrete_sequence=["#4a90d9"],
    )
    fig.add_vline(x=mean_v, line_dash="dash", line_color="red",
                  annotation_text=f"μ={mean_v:.2f}")
    fig.add_vline(x=mean_v + std_v, line_dash="dot", line_color="orange",
                  annotation_text=f"+σ")
    fig.add_vline(x=mean_v - std_v, line_dash="dot", line_color="orange",
                  annotation_text=f"-σ")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"🍕 Food delivery: {food_ex}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLT
# ═══════════════════════════════════════════════════════════════════════════════
st.header("2. Центральная предельная теорема (CLT)")

with st.expander("📖 Теория", expanded=True):
    st.markdown("""
**Теорема:** Если брать достаточно большие выборки из **любого** распределения,
распределение выборочных средних будет стремиться к **нормальному**:

$$\\bar{X}_n \\xrightarrow{d} N\\left(\\mu, \\frac{\\sigma^2}{n}\\right)$$

**Что это значит практически:**
- Мы не знаем распределение данных (заказы, время, чеки) — не важно
- Средние по достаточно большим выборкам всегда нормально распределены
- Это основа для всех t-тестов, z-тестов, доверительных интервалов
- В food delivery: среднее число заказов за неделю ~ Normal, даже если за день ~ Poisson
""")

col_clt1, col_clt2 = st.columns([1, 3])

with col_clt1:
    source_dist = st.selectbox(
        "Исходное распределение",
        ["Экспоненциальное (скошенное)", "Равномерное", "Пуассона"],
        key="clt_dist",
    )
    sample_size = st.select_slider(
        "Размер каждой выборки (n)",
        options=[1, 2, 5, 10, 30, 50, 100],
        value=30,
    )
    n_experiments = st.slider("Число экспериментов", 200, 3000, 1000, step=200)

rng = np.random.default_rng(0)
sample_means = []

for _ in range(n_experiments):
    if source_dist == "Экспоненциальное (скошенное)":
        s = rng.exponential(1.0, sample_size)
    elif source_dist == "Равномерное":
        s = rng.uniform(0, 10, sample_size)
    else:
        s = rng.poisson(3, sample_size).astype(float)
    sample_means.append(np.mean(s))

sample_means = np.array(sample_means)

# Theoretical normal
mu_th  = np.mean(sample_means)
std_th = np.std(sample_means)
x_range = np.linspace(mu_th - 4*std_th, mu_th + 4*std_th, 200)
normal_pdf = stats.norm.pdf(x_range, mu_th, std_th)

with col_clt2:
    fig_clt = go.Figure()
    fig_clt.add_trace(go.Histogram(
        x=sample_means, nbinsx=50, histnorm="probability density",
        name="Выборочные средние",
        marker_color="#4a90d9", opacity=0.7,
    ))
    fig_clt.add_trace(go.Scatter(
        x=x_range, y=normal_pdf, mode="lines",
        name="Нормальное (теор.)",
        line=dict(color="red", width=2),
    ))
    w_stat, p_shapiro = stats.shapiro(sample_means[:500])
    fig_clt.update_layout(
        title=f"CLT: n={sample_size}, {n_experiments} экспериментов | "
              f"Shapiro-Wilk p={p_shapiro:.3f}",
        xaxis_title="Выборочное среднее",
        yaxis_title="Плотность",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_clt, use_container_width=True)

    if p_shapiro > 0.05:
        st.success(f"✅ При n={sample_size} распределение средних уже нормальное (p={p_shapiro:.3f} > 0.05)")
    else:
        st.warning(f"⚠️ При n={sample_size} ещё не нормальное. Увеличь размер выборки.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
st.header("3. Доверительные интервалы")

with st.expander("📖 Теория", expanded=True):
    st.markdown("""
**Доверительный интервал (CI)** 95% не значит «с вероятностью 95% истинное значение в этом интервале».

**Правильное понимание:** Если повторить эксперимент 100 раз и построить 100 CI,
примерно **95 из них** будут содержать истинное значение.

Формула для CI среднего (при известной σ):

$$\\bar{X} \\pm z_{\\alpha/2} \\cdot \\frac{\\sigma}{\\sqrt{n}}$$

Для 95%: $z_{0.025} = 1.96$

**Что влияет на ширину CI:**
- Больше $n$ → уже CI ✅
- Меньше σ → уже CI ✅
- Выше уровень доверия → шире CI
""")

col_ci1, col_ci2 = st.columns([1, 3])

with col_ci1:
    true_mean  = st.slider("Истинное среднее (μ)", 0.0, 20.0, 10.0, step=0.5)
    true_std   = st.slider("Истинное std (σ)", 0.5, 10.0, 3.0, step=0.5)
    ci_n       = st.slider("Размер выборки (n)", 5, 200, 30)
    ci_level   = st.select_slider("Уровень доверия", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
    n_ci_runs  = st.slider("Число повторных экспериментов", 20, 100, 50)

rng = np.random.default_rng(7)
z = stats.norm.ppf((1 + ci_level) / 2)
ci_results = []

for i in range(n_ci_runs):
    sample = rng.normal(true_mean, true_std, ci_n)
    x_bar  = np.mean(sample)
    margin = z * true_std / np.sqrt(ci_n)
    lo, hi = x_bar - margin, x_bar + margin
    ci_results.append({"run": i, "mean": x_bar, "lo": lo, "hi": hi,
                        "covers": lo <= true_mean <= hi})

df_ci = pd.DataFrame(ci_results)
coverage = df_ci["covers"].mean() * 100

with col_ci2:
    fig_ci = go.Figure()
    for _, row in df_ci.iterrows():
        color = "#4caf50" if row["covers"] else "#f44336"
        fig_ci.add_trace(go.Scatter(
            x=[row["lo"], row["hi"]], y=[row["run"], row["run"]],
            mode="lines", line=dict(color=color, width=2), showlegend=False,
        ))
        fig_ci.add_trace(go.Scatter(
            x=[row["mean"]], y=[row["run"]],
            mode="markers", marker=dict(color=color, size=5), showlegend=False,
        ))
    fig_ci.add_vline(x=true_mean, line_dash="dash", line_color="black",
                     annotation_text=f"μ={true_mean}", annotation_position="top right")
    fig_ci.update_layout(
        title=f"{ci_level*100:.0f}% CI — покрытие: {coverage:.0f}% "
              f"({'✅' if abs(coverage - ci_level*100) < 10 else '⚠️'})",
        xaxis_title="Значение", yaxis_title="Эксперимент",
        height=max(350, n_ci_runs * 6),
        showlegend=False,
    )
    st.plotly_chart(fig_ci, use_container_width=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    ci_width = 2 * z * true_std / np.sqrt(ci_n)
    st.metric("Ширина CI", f"±{ci_width/2:.2f}")
with col_m2:
    st.metric("Покрытие", f"{coverage:.0f}%", delta=f"{coverage-ci_level*100:+.0f}%")
with col_m3:
    st.metric("Не накрыли μ", f"{int((1-df_ci['covers'].mean())*n_ci_runs)} / {n_ci_runs}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════════════════
st.header("4. Проверка гипотез и p-value")

with st.expander("📖 Теория", expanded=True):
    st.markdown("""
**Нулевая гипотеза H₀** — "ничего не произошло": разницы нет, эффекта нет.

**Альтернативная гипотеза H₁** — "есть эффект".

**p-value** — вероятность получить такой же или более экстремальный результат, **если H₀ верна**.

> p-value — это **НЕ** вероятность того что H₀ верна.

**Ошибки:**

| | H₀ верна | H₀ неверна |
|--|----------|-----------|
| Отвергаем H₀ | ❌ Ошибка I рода (α) | ✅ Верное решение |
| Не отвергаем H₀ | ✅ Верное решение | ❌ Ошибка II рода (β) |

- **α** (уровень значимости) — допустимая вероятность ошибки I рода. Обычно 0.05.
- **Мощность** (1-β) — вероятность найти эффект если он есть.

**В food delivery:** H₀ = "промо не влияет на заказы". Отвергаем H₀ при p < 0.05.
""")

col_h1, col_h2 = st.columns([1, 3])

with col_h1:
    h0_mean    = st.number_input("H₀: μ₀", value=10.0, step=0.5)
    true_mu    = st.slider("Истинное среднее", 8.0, 14.0, 11.5, step=0.1)
    h_std      = st.slider("σ (известная)", 0.5, 5.0, 2.0, step=0.1)
    h_n        = st.slider("Размер выборки", 5, 300, 50)
    alpha      = st.select_slider("α (уровень значимости)", [0.01, 0.05, 0.10], value=0.05)
    n_sim_h    = st.slider("Число симуляций", 100, 2000, 500, step=100)

rng  = np.random.default_rng(13)
null_stats = rng.normal(h0_mean, h_std / np.sqrt(h_n), 2000)
alt_data   = [np.mean(rng.normal(true_mu, h_std, h_n)) for _ in range(n_sim_h)]

z_crit = stats.norm.ppf(1 - alpha / 2)
se = h_std / np.sqrt(h_n)

x_range = np.linspace(h0_mean - 4*se, true_mu + 4*se, 500)
null_pdf = stats.norm.pdf(x_range, h0_mean, se)
alt_pdf  = stats.norm.pdf(x_range, true_mu, se)

with col_h2:
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=x_range, y=null_pdf, mode="lines", name="H₀ распределение",
        line=dict(color="#4a90d9", width=2), fill="tozeroy",
        fillcolor="rgba(74,144,217,0.15)",
    ))
    fig_h.add_trace(go.Scatter(
        x=x_range, y=alt_pdf, mode="lines", name="Истинное распределение",
        line=dict(color="#f44336", width=2), fill="tozeroy",
        fillcolor="rgba(244,67,54,0.15)",
    ))

    lo_crit = h0_mean - z_crit * se
    hi_crit = h0_mean + z_crit * se

    # Rejection region
    x_rej_r = x_range[x_range > hi_crit]
    fig_h.add_trace(go.Scatter(
        x=np.concatenate([[hi_crit], x_rej_r, [x_rej_r[-1]]]),
        y=np.concatenate([[0], stats.norm.pdf(x_rej_r, h0_mean, se), [0]]),
        fill="toself", fillcolor="rgba(255,100,0,0.3)",
        line=dict(color="rgba(0,0,0,0)"), name="Зона отклонения (α/2)",
    ))

    fig_h.add_vline(x=h0_mean,  line_dash="dash", line_color="#4a90d9",
                    annotation_text="μ₀")
    fig_h.add_vline(x=true_mu,  line_dash="dash", line_color="#f44336",
                    annotation_text="μ₁")
    fig_h.add_vline(x=hi_crit,  line_dash="dot",  line_color="orange",
                    annotation_text=f"z-crit={z_crit:.2f}")

    effect = true_mu - h0_mean
    power = 1 - stats.norm.cdf(hi_crit, true_mu, se)

    fig_h.update_layout(
        title=f"H₀: μ={h0_mean}  vs  H₁: μ={true_mu}  |  Мощность = {power:.2%}",
        xaxis_title="Выборочное среднее",
        yaxis_title="Плотность",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_h, use_container_width=True)

col_n1, col_n2, col_n3, col_n4 = st.columns(4)
with col_n1:
    st.metric("Эффект (μ₁ − μ₀)", f"{true_mu - h0_mean:+.1f}")
with col_n2:
    st.metric("Критическое значение", f"±{z_crit:.2f}σ")
with col_n3:
    st.metric("Мощность (1−β)", f"{power:.1%}",
              delta="✅ хорошо" if power >= 0.8 else "⚠️ мало",
              delta_color="off")
with col_n4:
    min_n = int(np.ceil(((z_crit + stats.norm.ppf(0.8)) * h_std / (true_mu - h0_mean)) ** 2)) if true_mu != h0_mean else "∞"
    st.metric("Мин. n для 80% мощности", min_n)

st.info("🍕 Food delivery: тестируем промо. H₀ = промо не влияет. "
        f"При μ₀={h0_mean}, истинном эффекте +{true_mu-h0_mean:.1f} и n={h_n} — "
        f"мощность теста {power:.0%}.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CODE
# ═══════════════════════════════════════════════════════════════════════════════
st.header("💻 Python-код")
with st.expander("Показать код"):
    st.code("""
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 1. Генерация распределений
orders_per_hour = rng.poisson(lam=8, size=1000)     # заказов в час
delivery_time   = rng.normal(loc=30, scale=8, size=1000)  # минут

print(f"Заказов/час: mean={orders_per_hour.mean():.1f}, std={orders_per_hour.std():.1f}")
print(f"Время доставки: mean={delivery_time.mean():.1f}, std={delivery_time.std():.1f}")

# 2. CLT: среднее за неделю → Normal
weekly_means = [rng.poisson(8, 7*24).mean() for _ in range(1000)]  # 7 дней × 24 часа
stat, p = stats.shapiro(weekly_means[:500])
print(f"Shapiro-Wilk p={p:.3f} — {'нормальное ✅' if p > 0.05 else 'не нормальное'}")

# 3. Доверительный интервал для среднего
sample = rng.normal(30, 8, 50)  # 50 доставок
mean, std, n = sample.mean(), sample.std(ddof=1), len(sample)
ci_lo, ci_hi = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
print(f"95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]")

# 4. Двухвыборочный t-тест (A/B промо)
control   = rng.normal(10, 3, 200)  # контроль
treatment = rng.normal(11.5, 3, 200)  # тест
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"t={t_stat:.2f}, p={p_value:.4f} → {'отвергаем H₀ ✅' if p_value < 0.05 else 'не отвергаем H₀'}")
""", language="python")

st.divider()
st.header("✅ Главное")
st.success("""
1. **Нормальное** — ошибки и непрерывные метрики. **Пуассона** — счётные данные (заказы/час). **Биномиальное** — доли.
2. **CLT**: средние по выборкам любого распределения → Normal при достаточном n.
3. **Доверительный интервал**: 95% CI — не «95% что μ здесь», а «95% таких CI содержат μ».
4. **p-value** — вероятность такого результата если H₀ верна. НЕ вероятность что H₀ верна.
5. **Мощность** (1−β) ≥ 0.8 — считай нужный размер выборки ДО эксперимента.
""")
render_nav(current=1)
