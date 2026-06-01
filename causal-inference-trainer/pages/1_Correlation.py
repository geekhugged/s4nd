import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from utils.generators import ice_cream_drowning, couriers_vs_orders

st.set_page_config(page_title="Корреляция vs Причинность", page_icon="🔗", layout="wide")

# ── THEORY ────────────────────────────────────────────────────────────────────
st.title("🔗 Модуль 1: Корреляция vs Причинность")

st.markdown("""
## Теория

### Что такое корреляция?
Корреляция Пирсона измеряет **линейную связь** между двумя переменными:

$$r = \\frac{\\sum(X_i - \\bar{X})(Y_i - \\bar{Y})}{\\sqrt{\\sum(X_i-\\bar{X})^2 \\cdot \\sum(Y_i-\\bar{Y})^2}}$$

- $r = 1$ — идеальная положительная связь
- $r = 0$ — нет линейной связи
- $r = -1$ — идеальная отрицательная связь

### Что такое причинность?
$X$ **вызывает** $Y$ если при принудительном изменении $X$ (вмешательстве) значение $Y$ изменяется.

Нотация: $P(Y \\mid do(X=x))$ — распределение $Y$ при вмешательстве, а не просто наблюдении.

### Почему корреляция ≠ причинность?

| Причина | Описание | Пример |
|---------|----------|--------|
| **Конфаундер** | Третья переменная $Z$ влияет на оба: $X \\leftarrow Z \\rightarrow Y$ | Лето → мороженое и утопления |
| **Обратная причинность** | $Y$ вызывает $X$, не наоборот | Больше полицейских → больше преступлений? |
| **Случайность** | При многих переменных ложные корреляции неизбежны | Потребление сыра ~ смерти в постели |

### Ключевой вопрос
> *"Что случится с $Y$ если мы принудительно изменим $X$?"*
""")

st.divider()

# ── SIMULATION 1: ICE CREAM ───────────────────────────────────────────────────
st.header("🧊 Симуляция 1: Мороженое и утопления")
st.markdown(
    "Продажи мороженого **коррелируют** с числом утоплений. "
    "Причина — **лето (конфаундер)**. Уберём конфаундер — корреляция исчезнет."
)

col_ctrl, col_plot = st.columns([1, 3])
with col_ctrl:
    n = st.slider("Размер выборки", 50, 1000, 300, key="ice_n")
    strength = st.slider("Сила конфаундера (сезонность)", 0.0, 5.0, 3.0, step=0.5)
    show_conf = st.checkbox("Показать конфаундер (сезон)", value=False)

df_ice = ice_cream_drowning(n=n, confounder_strength=strength)
r_ice, _ = stats.pearsonr(df_ice["ice_cream_sales"], df_ice["drowning_incidents"])

with col_plot:
    if show_conf:
        fig = px.scatter(
            df_ice, x="ice_cream_sales", y="drowning_incidents",
            color="season_label",
            color_discrete_map={"Лето": "#ff6b35", "Зима": "#4cc9f0"},
            labels={"ice_cream_sales": "Продажи мороженого", "drowning_incidents": "Утопления",
                    "season_label": "Сезон"},
            trendline="ols",
            title=f"r = {r_ice:.2f} — конфаундер (сезон) объясняет всё",
        )
    else:
        fig = px.scatter(
            df_ice, x="ice_cream_sales", y="drowning_incidents",
            labels={"ice_cream_sales": "Продажи мороженого", "drowning_incidents": "Утопления"},
            trendline="ols",
            title=f"Корреляция r = {r_ice:.2f} — выглядит как причинность",
        )
    st.plotly_chart(fig, use_container_width=True)

st.info(f"r = **{r_ice:.2f}**. Поставь «Показать конфаундер» — и увидишь что оба ряда просто растут летом.")

st.divider()

# ── SIMULATION 2: FOOD DELIVERY ────────────────────────────────────────────────
st.header("🍕 Симуляция 2: Курьеры и заказы")
st.markdown("""
**Наблюдение:** в часы когда онлайн больше курьеров — больше заказов.

**Ошибочный вывод:** нанять ещё 100 курьеров в 3 ночи → заказов станет больше.

**Реальность:** оба зависят от **времени суток** — конфаундер.
""")

col_ctrl2, col_plot2 = st.columns([1, 3])
with col_ctrl2:
    n2 = st.slider("Размер выборки", 50, 1000, 400, key="del_n")
    show_tod = st.checkbox("Показать время суток", value=False)

df_del = couriers_vs_orders(n=n2)
r_del, _ = stats.pearsonr(df_del["couriers"], df_del["orders"])

with col_plot2:
    if show_tod:
        fig2 = px.scatter(
            df_del, x="couriers", y="orders", color="period",
            color_discrete_map={"Утро": "#f9c74f", "День": "#43aa8b", "Вечер": "#f3722c"},
            labels={"couriers": "Курьеров онлайн", "orders": "Заказов", "period": "Время суток"},
            trendline="ols",
            title=f"r = {r_del:.2f} — конфаундер: пиковые часы",
        )
    else:
        fig2 = px.scatter(
            df_del, x="couriers", y="orders",
            labels={"couriers": "Курьеров онлайн", "orders": "Заказов"},
            trendline="ols",
            title=f"Корреляция r = {r_del:.2f} — нанять больше курьеров?",
        )
    st.plotly_chart(fig2, use_container_width=True)

st.warning(
    "Наём курьеров в 3 ночи не увеличит заказы. "
    "Правильный вопрос: *при фиксированном времени суток*, "
    "увеличение числа курьеров влияет на заказы?"
)

st.divider()

# ── PYTHON CODE ───────────────────────────────────────────────────────────────
st.header("💻 Python: как это проверить")

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

# Обе переменные зависят от конфаундера
couriers = 50 + 80 * peak + rng.normal(0, 8, n)
orders   = 100 + 200 * peak + rng.normal(0, 20, n)

# Наивная корреляция — высокая!
r_naive, _ = stats.pearsonr(couriers, orders)
print(f"Наивная корреляция: {r_naive:.2f}")        # ~0.93

# Частичная корреляция — убираем конфаундер через остатки регрессии
def residuals(x, z):
    coef = np.polyfit(z, x, 1)
    return x - (coef[0] * z + coef[1])

c_resid = residuals(couriers, peak)
o_resid = residuals(orders, peak)

r_partial, _ = stats.pearsonr(c_resid, o_resid)
print(f"Частичная корреляция: {r_partial:.2f}")    # ~0.0 — связи нет!
""", language="python")

st.divider()

# ── TAKEAWAYS ─────────────────────────────────────────────────────────────────
st.header("✅ Главное из модуля")
st.success("""
1. **Корреляция** — мера совместного изменения. Не объясняет почему.
2. **Конфаундер** — скрытая третья переменная, создающая ложную корреляцию.
3. **Вмешательство** (do-operator) — ключ к причинности. "Что будет с Y если мы принудительно изменим X?"
4. **В food delivery**: большинство операционных гипотез строятся на наблюдательных данных. Без causal inference — высокий риск неверных решений.
""")

st.markdown("**👉 Следующий модуль:** Potential Outcomes →")
