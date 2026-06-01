import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.generators import promo_with_selection_bias
from utils.ui import render_sidebar, render_progress, render_nav

st.set_page_config(page_title="Potential Outcomes", page_icon="🎯", layout="wide")
render_sidebar(current=3)

st.title("🎯 Модуль 2: Potential Outcomes")
render_progress(current=3)
st.divider()

# ── THEORY ────────────────────────────────────────────────────────────────────
with st.expander("📖 Теория", expanded=True):
    st.markdown("""
### Модель Рубина (Rubin Causal Model)

Для каждого пользователя $i$ существуют **два потенциальных исхода**:
- $Y_i(1)$ — исход **с** воздействием (промо, фича, лечение)
- $Y_i(0)$ — исход **без** воздействия

**Индивидуальный эффект:** $\\tau_i = Y_i(1) - Y_i(0)$

---

### Фундаментальная проблема причинного вывода
> Мы никогда не наблюдаем оба исхода одновременно.

$$Y_i^{obs} = T_i \\cdot Y_i(1) + (1 - T_i) \\cdot Y_i(0)$$

Второй исход — **контрфактуал** — навсегда скрыт.

---

### Что мы оцениваем?

| Метрика | Формула | Смысл |
|---------|---------|-------|
| **ATE** | $E[Y(1) - Y(0)]$ | Средний эффект по всей популяции |
| **ATT** | $E[Y(1) - Y(0) \\mid T=1]$ | Эффект среди получивших воздействие |
| **ATU** | $E[Y(1) - Y(0) \\mid T=0]$ | Эффект среди не получивших |

---

### Selection Bias — главная ловушка

Наивная оценка: $\\hat{\\tau}_{naive} = E[Y \\mid T=1] - E[Y \\mid T=0]$

$$= \\underbrace{\\text{ATT}}_{\\text{истинный эффект}} + \\underbrace{E[Y(0)\\mid T=1] - E[Y(0)\\mid T=0]}_{\\text{Selection Bias}}$$

Если промо дают активным пользователям → наивная оценка **завышена**.
""")

st.divider()

# ── SIMULATION ────────────────────────────────────────────────────────────────
st.header("🍕 Симуляция: Эффект промокода")
st.markdown(
    "Промо рассылается чаще активным пользователям. "
    "**Истинный эффект** зафиксирован — посмотрим насколько наивная оценка ошибается."
)

c1, c2 = st.columns([1, 3])
with c1:
    n = st.slider("Пользователей", 200, 2000, 1000, step=100)
    true_effect = st.slider("Истинный эффект (заказов)", 0.0, 10.0, 2.0, step=0.5)
    show_cf = st.checkbox("Показать контрфактуалы")

df = promo_with_selection_bias(n=n, true_effect=true_effect)
naive_ate = df[df.promo == 1]["orders"].mean() - df[df.promo == 0]["orders"].mean()
true_ate  = (df["y1_true"] - df["y0_true"]).mean()
bias      = naive_ate - true_ate

with c2:
    tab1, tab2, tab3 = st.tabs(["📊 Распределение", "🎯 Контрфактуалы", "📈 Selection Bias"])

    with tab1:
        fig1 = px.histogram(
            df, x="orders",
            color=df["promo"].map({1: "Получили промо", 0: "Не получили промо"}),
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Получили промо": "#4caf50", "Не получили промо": "#ff5722"},
            labels={"orders": "Заказов за неделю", "color": "Группа"},
            title="Распределение заказов по группам",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        if show_cf:
            sample = df.sample(min(150, n), random_state=42).sort_values("activity")
            fig2 = go.Figure()
            for _, row in sample.iterrows():
                fig2.add_trace(go.Scatter(
                    x=[row["activity"], row["activity"]],
                    y=[row["y0_true"], row["y1_true"]],
                    mode="lines", line=dict(color="#e0e0e0", width=1), showlegend=False,
                ))
            fig2.add_trace(go.Scatter(
                x=sample["activity"], y=sample["y0_true"], mode="markers",
                name="Y(0) — без промо", marker=dict(color="#ff5722", size=5),
            ))
            fig2.add_trace(go.Scatter(
                x=sample["activity"], y=sample["y1_true"], mode="markers",
                name="Y(1) — с промо", marker=dict(color="#4caf50", size=5),
            ))
            fig2.update_layout(
                title="Потенциальные исходы Y(1) и Y(0) для каждого пользователя",
                xaxis_title="Активность пользователя",
                yaxis_title="Заказов",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("В реальности для каждого пользователя мы видим только одну точку.")
        else:
            st.info("Включи «Показать контрфактуалы» — увидишь оба потенциальных исхода "
                    "(нереальный сценарий, доступен только в симуляции).")

    with tab3:
        activity_bins = pd.cut(df["activity"], bins=8, labels=False)
        grp = df.groupby([activity_bins, "promo"])["orders"].mean().unstack()
        grp.columns = ["Без промо", "С промо"]
        grp = grp.dropna().reset_index()
        grp["Децил активности"] = grp["activity"].astype(int).astype(str)
        fig3 = px.bar(
            grp.melt(id_vars="Децил активности", value_vars=["Без промо", "С промо"],
                     var_name="Группа", value_name="Заказов"),
            x="Децил активности", y="Заказов", color="Группа", barmode="group",
            color_discrete_map={"С промо": "#4caf50", "Без промо": "#ff5722"},
            title="Активные пользователи чаще получают промо и делают больше заказов",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📐 Итог симуляции")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Истинный ATE", f"+{true_ate:.2f}", help="E[Y(1)−Y(0)] — знаем только в симуляции")
with m2:
    st.metric("Наивная оценка", f"+{naive_ate:.2f}",
              delta=f"{bias:+.2f} смещение", delta_color="inverse")
with m3:
    st.metric("Selection Bias", f"{bias:+.2f}",
              help="На сколько наивная оценка завышает истинный эффект")

if bias > 0.5:
    st.error(
        f"Наивная оценка завышает эффект промо на **{bias:.1f} заказа** "
        f"({bias/true_ate*100:.0f}% от истинного значения). "
        "Это и есть selection bias — промо получают изначально более активные пользователи."
    )

st.divider()

# ── CODE ──────────────────────────────────────────────────────────────────────
st.header("💻 Python-код")
with st.expander("Показать код"):
    st.code("""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

# Скрытый конфаундер: историческая активность
activity = rng.normal(0, 1, n)

# Промо дают чаще активным пользователям
promo_prob = 1 / (1 + np.exp(-activity))
promo = rng.binomial(1, promo_prob, n)

# Потенциальные исходы (истинный эффект = +2)
y0 = np.clip(5 + 3 * activity + rng.normal(0, 1, n), 0, None)
y1 = y0 + 2

# Наблюдаемый исход
y_obs = promo * y1 + (1 - promo) * y0

df = pd.DataFrame({"promo": promo, "orders": y_obs, "y0": y0, "y1": y1})

# Наивная оценка — ЗАВЫШЕНА из-за selection bias
naive = df[df.promo==1]["orders"].mean() - df[df.promo==0]["orders"].mean()
print(f"Наивная оценка ATE: {naive:.2f}")    # ~4.5

# Истинный ATE (только в симуляции)
true_ate = (df["y1"] - df["y0"]).mean()
print(f"Истинный ATE:       {true_ate:.2f}") # ~2.0

# Решение: рандомизация (A/B тест), matching, DiD, IV
""", language="python")

st.divider()
st.header("✅ Главное")
st.success("""
1. У каждого пользователя есть два потенциальных исхода — наблюдается только один.
2. Контрфактуал — то что было бы при другом решении. Всегда скрыт.
3. ATE = E[Y(1) − Y(0)] — средний каузальный эффект.
4. Selection bias возникает когда воздействие назначается не случайно.
5. В food delivery: пользователи с промо уже более активны → наивная оценка завышает эффект.
""")
render_nav(current=3)
