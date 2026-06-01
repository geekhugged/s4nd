import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.generators import promo_with_selection_bias

st.set_page_config(page_title="Potential Outcomes", page_icon="🎯", layout="wide")

st.title("🎯 Модуль 2: Potential Outcomes")

# ── THEORY ────────────────────────────────────────────────────────────────────
st.markdown("""
## Теория

### Модель Рубина (Rubin Causal Model)

Для каждого пользователя $i$ существуют **два потенциальных исхода**:

- $Y_i(1)$ — что случится если пользователь **получит** воздействие (промо, лечение, фичу)
- $Y_i(0)$ — что случится если пользователь **не получит** воздействие

**Индивидуальный эффект:** $\\tau_i = Y_i(1) - Y_i(0)$

### Фундаментальная проблема причинного вывода

> Мы никогда не можем наблюдать оба исхода для одного человека одновременно.

Для каждого $i$ наблюдается только один исход:

$$Y_i^{obs} = T_i \\cdot Y_i(1) + (1 - T_i) \\cdot Y_i(0)$$

Второй исход — **контрфактуал** — навсегда скрыт.

### Что мы оцениваем?

| Метрика | Формула | Смысл |
|---------|---------|-------|
| **ATE** (Average Treatment Effect) | $E[Y(1) - Y(0)]$ | Средний эффект по всей популяции |
| **ATT** (Average Treatment on Treated) | $E[Y(1) - Y(0) \\mid T=1]$ | Эффект среди тех кто получил воздействие |
| **ATU** (Average Treatment on Untreated) | $E[Y(1) - Y(0) \\mid T=0]$ | Эффект среди тех кто не получил |

### Selection Bias — главная ловушка

Наивная оценка: $\\hat{\\tau}_{naive} = E[Y \\mid T=1] - E[Y \\mid T=0]$

Это **не ATE**! Разложим:

$$E[Y \\mid T=1] - E[Y \\mid T=0] = \\underbrace{E[Y(1)-Y(0) \\mid T=1]}_{ATT} + \\underbrace{E[Y(0) \\mid T=1] - E[Y(0) \\mid T=0]}_{\\text{Selection Bias}}$$

Если промо дают более активным пользователям — наивная оценка будет **завышена**.
""")

st.divider()

# ── SIMULATION ────────────────────────────────────────────────────────────────
st.header("🍕 Симуляция: Эффект промокода")
st.markdown("""
Компания рассылает промокод. Более активные пользователи получают его с большей вероятностью.
**Истинный эффект промо** = +2 заказа. Посмотрим что покажет наивное сравнение.
""")

col_ctrl, col_main = st.columns([1, 3])
with col_ctrl:
    n = st.slider("Пользователей", 200, 2000, 1000, step=100)
    true_effect = st.slider("Истинный эффект промо (заказов)", 0.0, 10.0, 2.0, step=0.5)
    show_cf = st.checkbox("Показать контрфактуалы (нереальный сценарий)", value=False)

df = promo_with_selection_bias(n=n, true_effect=true_effect)

naive_ate = df[df.promo == 1]["orders"].mean() - df[df.promo == 0]["orders"].mean()
true_ate = (df["y1_true"] - df["y0_true"]).mean()

with col_main:
    tab1, tab2, tab3 = st.tabs(["📊 Распределение заказов", "🎯 Контрфактуалы", "📈 Selection Bias"])

    with tab1:
        fig = px.histogram(
            df, x="orders", color=df["promo"].map({1: "Получили промо", 0: "Не получили промо"}),
            barmode="overlay", opacity=0.7,
            labels={"orders": "Заказов", "color": "Группа"},
            title="Распределение заказов по группам",
            color_discrete_map={"Получили промо": "#4caf50", "Не получили промо": "#ff5722"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if show_cf:
            sample = df.sample(min(200, n), random_state=42).sort_values("activity")
            fig2 = go.Figure()
            for _, row in sample.iterrows():
                fig2.add_trace(go.Scatter(
                    x=[row["activity"], row["activity"]],
                    y=[row["y0_true"], row["y1_true"]],
                    mode="lines", line=dict(color="lightgray", width=1), showlegend=False,
                ))
            fig2.add_trace(go.Scatter(
                x=sample["activity"], y=sample["y0_true"],
                mode="markers", name="Y(0) — без промо",
                marker=dict(color="#ff5722", size=5),
            ))
            fig2.add_trace(go.Scatter(
                x=sample["activity"], y=sample["y1_true"],
                mode="markers", name="Y(1) — с промо",
                marker=dict(color="#4caf50", size=5),
            ))
            fig2.update_layout(
                title="Потенциальные исходы: Y(1) vs Y(0) для каждого пользователя",
                xaxis_title="Активность пользователя",
                yaxis_title="Заказов",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("В реальности мы видим только одну точку на каждого пользователя. Здесь — нереальный сценарий «знаем оба исхода».")
        else:
            st.info("Включи «Показать контрфактуалы» чтобы увидеть оба потенциальных исхода для каждого пользователя (нереальный сценарий).")

    with tab3:
        activity_bins = pd.cut(df["activity"], bins=10)
        bias_df = df.groupby([activity_bins, "promo"])["orders"].mean().unstack()
        bias_df.columns = ["Без промо", "С промо"]
        bias_df.index = [str(i) for i in bias_df.index]
        fig3 = px.bar(
            bias_df.reset_index().melt(id_vars="activity", var_name="Группа", value_name="Заказов"),
            x="activity", y="Заказов", color="Группа", barmode="group",
            color_discrete_map={"С промо": "#4caf50", "Без промо": "#ff5722"},
            title="Заказы по децилям активности — активные пользователи чаще получают промо",
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📐 Результаты")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Истинный ATE", f"+{true_ate:.2f} заказов", help="E[Y(1) - Y(0)] — знаем только в симуляции")
with m2:
    st.metric("Наивная оценка", f"+{naive_ate:.2f} заказов",
              delta=f"{naive_ate - true_ate:+.2f} (смещение)",
              delta_color="inverse")
with m3:
    bias = naive_ate - true_ate
    st.metric("Selection Bias", f"{bias:+.2f} заказов",
              help="Насколько наивная оценка завышает истинный эффект")

if naive_ate > true_ate + 0.3:
    st.error(
        f"Наивная оценка (+{naive_ate:.1f}) **завышает** истинный эффект (+{true_ate:.1f}) "
        f"на **{naive_ate - true_ate:.1f} заказов**. "
        "Причина: промо получают более активные пользователи."
    )

st.divider()

# ── CODE ──────────────────────────────────────────────────────────────────────
st.header("💻 Python: potential outcomes на практике")

with st.expander("Показать код"):
    st.code("""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

# Скрытый конфаундер: активность пользователя
activity = rng.normal(0, 1, n)

# Промо дают чаще активным пользователям
promo_prob = 1 / (1 + np.exp(-activity))
promo = rng.binomial(1, promo_prob, n)

# Потенциальные исходы
y0 = np.clip(5 + 3 * activity + rng.normal(0, 1, n), 0, None)
y1 = y0 + 2  # истинный эффект = +2 заказа

# Наблюдаемый исход
y_obs = promo * y1 + (1 - promo) * y0

df = pd.DataFrame({"promo": promo, "orders": y_obs,
                   "y0": y0, "y1": y1, "activity": activity})

# Наивная оценка — НЕВЕРНА из-за selection bias
naive = df[df.promo==1]["orders"].mean() - df[df.promo==0]["orders"].mean()
print(f"Наивная оценка ATE: {naive:.2f}")     # ~4.5 — завышена!

# Истинный ATE (доступен только в симуляции)
true_ate = (df["y1"] - df["y0"]).mean()
print(f"Истинный ATE:       {true_ate:.2f}")  # ~2.0

# Как исправить? → рандомизация, matching, DiD, IV (следующие модули)
""", language="python")

st.divider()

# ── TAKEAWAYS ─────────────────────────────────────────────────────────────────
st.header("✅ Главное из модуля")
st.success("""
1. **Потенциальные исходы** Y(1) и Y(0) существуют для каждого — но наблюдается только один.
2. **Контрфактуал** — то что было бы при другом решении. Никогда не наблюдается напрямую.
3. **ATE** = E[Y(1) - Y(0)] — средний каузальный эффект по популяции.
4. **Selection bias** возникает когда воздействие назначается не случайно.
5. В food delivery: пользователи получающие промо уже более активны → наивная оценка завышает эффект промо.
""")
st.markdown("**👉 Следующий модуль:** DAGs →")
