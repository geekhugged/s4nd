import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from utils.generators import ice_cream_drowning, couriers_vs_orders
from utils.ui import render_sidebar, render_progress, render_nav

st.set_page_config(page_title="Корреляция vs Причинность", page_icon="🔗", layout="wide")
render_sidebar(current=1)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔗 Модуль 1: Корреляция vs Причинность")
render_progress(current=1)
st.divider()

# ── THEORY ────────────────────────────────────────────────────────────────────
with st.expander("📖 Теория — раскрой перед симуляцией", expanded=True):
    st.markdown("""
### Корреляция
Коэффициент Пирсона измеряет **линейную связь** между двумя переменными:

$$r = \\frac{\\sum(X_i - \\bar{X})(Y_i - \\bar{Y})}{\\sqrt{\\sum(X_i-\\bar{X})^2 \\cdot \\sum(Y_i-\\bar{Y})^2}}$$

Диапазон: от −1 (идеальная отрицательная) до +1 (идеальная положительная). 0 — нет линейной связи.

---

### Причинность
$X$ **вызывает** $Y$ если при **принудительном изменении** $X$ (вмешательство, do-operator) значение $Y$ изменяется.

Нотация: $P(Y \\mid do(X=x))$ — отличается от обычного условного $P(Y \\mid X=x)$.

---

### Три причины ложных корреляций

| Причина | Структура | Пример |
|---------|-----------|--------|
| **Конфаундер** | $X \\leftarrow Z \\rightarrow Y$ | Лето → мороженое и утопления |
| **Обратная причинность** | $Y \\rightarrow X$ | Больше врачей → больше болезней? |
| **Случайность** | — | Потребление сыра ~ смерти в постели |

---

### Ключевой вопрос причинности
> *"Что случится с $Y$ если мы **принудительно изменим** $X$, удерживая всё остальное?"*
""")

st.divider()

# ── SIM 1: ICE CREAM ──────────────────────────────────────────────────────────
st.header("🧊 Симуляция 1: Мороженое и утопления")
st.markdown(
    "Продажи мороженого **коррелируют** с числом утоплений. "
    "Оба ряда растут летом — конфаундер создаёт ложную корреляцию."
)

c1, c2 = st.columns([1, 3])
with c1:
    n1 = st.slider("Размер выборки", 50, 1000, 300, key="s1_n")
    strength = st.slider("Сила сезонности", 0.0, 5.0, 3.0, step=0.5)
    show_conf = st.checkbox("Показать конфаундер (сезон)")

df1 = ice_cream_drowning(n=n1, confounder_strength=strength)
r1, _ = stats.pearsonr(df1["ice_cream_sales"], df1["drowning_incidents"])

with c2:
    if show_conf:
        fig1 = px.scatter(
            df1, x="ice_cream_sales", y="drowning_incidents", color="season_label",
            color_discrete_map={"Лето": "#ff6b35", "Зима": "#4cc9f0"},
            labels={"ice_cream_sales": "Продажи мороженого",
                    "drowning_incidents": "Утопления", "season_label": "Сезон"},
            trendline="ols", title=f"r = {r1:.2f} — конфаундер виден",
        )
    else:
        fig1 = px.scatter(
            df1, x="ice_cream_sales", y="drowning_incidents",
            labels={"ice_cream_sales": "Продажи мороженого", "drowning_incidents": "Утопления"},
            trendline="ols", title=f"r = {r1:.2f} — выглядит как причинность",
        )
    st.plotly_chart(fig1, use_container_width=True)

st.info(f"r = **{r1:.2f}**. Включи «Показать конфаундер» — оба ряда просто растут летом.")

st.divider()

# ── SIM 2: FOOD DELIVERY ──────────────────────────────────────────────────────
st.header("🍕 Симуляция 2: Курьеры и заказы")
st.markdown("""
**Наблюдение:** когда онлайн больше курьеров — больше заказов.
**Ошибочный вывод:** нанять ещё 100 курьеров → заказов станет больше.
**Реальность:** оба зависят от времени суток.
""")

c3, c4 = st.columns([1, 3])
with c3:
    n2 = st.slider("Размер выборки", 50, 1000, 400, key="s2_n")
    show_tod = st.checkbox("Показать время суток")

df2 = couriers_vs_orders(n=n2)
r2, _ = stats.pearsonr(df2["couriers"], df2["orders"])

with c4:
    if show_tod:
        fig2 = px.scatter(
            df2, x="couriers", y="orders", color="period",
            color_discrete_map={"Утро": "#f9c74f", "День": "#43aa8b", "Вечер": "#f3722c"},
            labels={"couriers": "Курьеров онлайн", "orders": "Заказов", "period": "Время суток"},
            trendline="ols", title=f"r = {r2:.2f} — конфаундер: пиковые часы",
        )
    else:
        fig2 = px.scatter(
            df2, x="couriers", y="orders",
            labels={"couriers": "Курьеров онлайн", "orders": "Заказов"},
            trendline="ols", title=f"r = {r2:.2f} — нанять больше курьеров?",
        )
    st.plotly_chart(fig2, use_container_width=True)

st.warning("Наём курьеров в 3 ночи не увеличит заказы. Конфаундер — время суток.")

st.divider()

# ── PARTIAL CORRELATION ───────────────────────────────────────────────────────
st.header("🔬 Partial Correlation: убираем конфаундер")
st.markdown("Контролируем время суток через остатки регрессии — корреляция исчезает.")

peak = (np.exp(-0.5 * ((df2["hour"] - 12) / 4) ** 2) +
        0.8 * np.exp(-0.5 * ((df2["hour"] - 19) / 2) ** 2))
c_resid = df2["couriers"] - np.polyval(np.polyfit(peak, df2["couriers"], 1), peak)
o_resid = df2["orders"]   - np.polyval(np.polyfit(peak, df2["orders"],   1), peak)
r_partial, _ = stats.pearsonr(c_resid, o_resid)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Наивная корреляция", f"{r2:.3f}", help="Включает влияние конфаундера")
with col_m2:
    st.metric("Partial correlation", f"{r_partial:.3f}",
              delta=f"{r_partial - r2:+.3f}",
              delta_color="off",
              help="После контроля времени суток")

fig3 = px.scatter(
    x=c_resid, y=o_resid,
    labels={"x": "Остатки (курьеры)", "y": "Остатки (заказы)"},
    trendline="ols",
    title=f"Partial correlation = {r_partial:.3f} — связи нет",
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── CODE ──────────────────────────────────────────────────────────────────────
st.header("💻 Python-код")
with st.expander("Показать код"):
    st.code("""
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
n = 500

# Конфаундер: пиковые часы
hour = rng.uniform(0, 24, n)
peak = np.exp(-0.5 * ((hour - 12) / 4) ** 2)

couriers = 50 + 80 * peak + rng.normal(0, 8, n)
orders   = 100 + 200 * peak + rng.normal(0, 20, n)

# Наивная корреляция — высокая!
r_naive, _ = stats.pearsonr(couriers, orders)
print(f"Наивная:         r = {r_naive:.2f}")   # ~0.93

# Partial correlation — убираем конфаундер
def residuals(x, z):
    coef = np.polyfit(z, x, 1)
    return x - np.polyval(coef, z)

r_partial, _ = stats.pearsonr(residuals(couriers, peak),
                               residuals(orders,   peak))
print(f"Partial corr:    r = {r_partial:.2f}")  # ~0.0
""", language="python")

st.divider()

# ── TAKEAWAYS ─────────────────────────────────────────────────────────────────
st.header("✅ Главное")
st.success("""
1. **Корреляция** — мера совместного изменения, не объясняет причину.
2. **Конфаундер** — скрытая третья переменная, создающая ложную корреляцию.
3. **Partial correlation** — убираем конфаундер через остатки регрессии.
4. **Вопрос причинности**: "Что будет с Y при принудительном изменении X?"
5. В food delivery: большинство операционных гипотез требуют проверки на конфаундеры.
""")

render_nav(current=1)
