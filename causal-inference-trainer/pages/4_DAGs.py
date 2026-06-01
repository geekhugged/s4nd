import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from scipy import stats
from utils.ui import render_sidebar, render_progress, render_nav

st.set_page_config(page_title="DAGs", page_icon="📊", layout="wide")
render_sidebar(current=4)

st.title("📊 Модуль 3: DAGs — Причинные Графы")
render_progress(current=4)
st.divider()

# ── THEORY ────────────────────────────────────────────────────────────────────
with st.expander("📖 Теория", expanded=True):
    st.markdown("""
### Что такое DAG?

**DAG** (Directed Acyclic Graph) — ориентированный граф без циклов:
- **Узлы** = переменные
- **Стрелки** = прямые причинные связи ($X \\rightarrow Y$: "$X$ причиняет $Y$")
- **Нет циклов** — причина не может быть своим следствием

DAG — это **декларация** наших причинно-следственных убеждений о мире.

---

### Три фундаментальные структуры

| Структура | Граф | Что происходит |
|-----------|------|----------------|
| **Chain (цепочка)** | $X \\rightarrow Z \\rightarrow Y$ | $Z$ — медиатор. Условие на $Z$ блокирует путь. |
| **Fork (развилка)** | $X \\leftarrow Z \\rightarrow Y$ | $Z$ — конфаундер. Создаёт ложную корреляцию X~Y. |
| **Collider (коллайдер)** | $X \\rightarrow Z \\leftarrow Y$ | $Z$ блокирует путь. Условие на $Z$ **открывает** ложную связь! |

---

### Backdoor Criterion

Чтобы идентифицировать $X \\rightarrow Y$, нужно заблокировать **все backdoor-пути** — пути из $X$ в $Y$, идущие "против" стрелки из $X$.

Набор $Z$ удовлетворяет backdoor criterion если:
1. Блокирует все backdoor-пути между $X$ и $Y$
2. Не содержит потомков $X$
""")

st.divider()

# ── THREE STRUCTURES ──────────────────────────────────────────────────────────
st.header("🔍 Три фундаментальные структуры")

def dag_fig(edges, pos, title, highlight=None, figsize=(3.5, 2.8)):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#ff6b35" if (highlight and n in highlight) else "#4a90d9"
              for n in G.nodes()]
    nx.draw_networkx(G, pos, ax=ax,
                     node_color=colors, node_size=1600,
                     font_color="white", font_size=11, font_weight="bold",
                     edge_color="#555", width=2, arrows=True, arrowsize=22,
                     connectionstyle="arc3,rad=0.08")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    return fig

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Цепочка")
    fig = dag_fig([("X","Z"),("Z","Y")],
                  {"X":(0,0),"Z":(1,0),"Y":(2,0)},
                  "X → Z → Y", highlight=["Z"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
**Z** — медиатор.

- Условие на **Z** блокирует путь X→Y
- *X ⊥ Y | Z*

Пример: Промо → Открытие приложения → Заказы
    """)

with c2:
    st.subheader("Развилка")
    fig = dag_fig([("Z","X"),("Z","Y")],
                  {"Z":(1,1),"X":(0,0),"Y":(2,0)},
                  "X ← Z → Y", highlight=["Z"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
**Z** — конфаундер.

- Создаёт ложную корреляцию X~Y
- Условие на **Z** блокирует
- *X ⊥ Y | Z*

Пример: Сезон → Заказы И Сезон → Курьеры
    """)

with c3:
    st.subheader("Коллайдер ⚠️")
    fig = dag_fig([("X","Z"),("Y","Z")],
                  {"X":(0,0),"Y":(2,0),"Z":(1,-1)},
                  "X → Z ← Y", highlight=["Z"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
**Z** — коллайдер.

- Путь **заблокирован** по умолчанию
- Условие на **Z** **открывает** ложную связь!

Пример: Навык + Удача → Нанят
    """)

st.divider()

# ── FOOD DELIVERY DAG ─────────────────────────────────────────────────────────
st.header("🍕 DAG для food delivery: промо → заказы")

c_dag, c_exp = st.columns([2, 1])

with c_dag:
    G = nx.DiGraph()
    edges = [("Activity","Promo"), ("Activity","Orders"),
             ("Promo","AppOpen"), ("AppOpen","Orders"),
             ("Promo","Orders"), ("Rain","Orders")]
    G.add_edges_from(edges)
    pos = {"Activity":(0,1), "Promo":(1,2),
           "AppOpen":(2.2,2), "Orders":(3,1), "Rain":(3,2.5)}
    node_colors = {"Activity":"#ff6b35","Promo":"#4a90d9",
                   "AppOpen":"#43aa8b","Orders":"#4a90d9","Rain":"#9b59b6"}

    fig_d, ax_d = plt.subplots(figsize=(7, 3.8))
    nx.draw_networkx(G, pos, ax=ax_d,
                     node_color=[node_colors[n] for n in G.nodes()],
                     node_size=2200, font_color="white", font_size=9.5, font_weight="bold",
                     edge_color="#555", width=2, arrows=True, arrowsize=20,
                     connectionstyle="arc3,rad=0.06")
    legend = [mpatches.Patch(color="#ff6b35", label="Конфаундер"),
              mpatches.Patch(color="#4a90d9", label="Переменные интереса"),
              mpatches.Patch(color="#43aa8b", label="Медиатор"),
              mpatches.Patch(color="#9b59b6", label="Инструмент (IV)")]
    ax_d.legend(handles=legend, loc="lower left", fontsize=8)
    ax_d.set_title("Causal DAG: Promo → Orders", fontsize=12, fontweight="bold")
    ax_d.axis("off")
    fig_d.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig_d, use_container_width=True)
    plt.close(fig_d)

with c_exp:
    st.markdown("""
**Backdoor-путь:**
`Promo ← Activity → Orders`

Чтобы найти истинный эффект промо → контролируй **Activity**.

---

**Медиатор AppOpen:**
Часть эффекта промо идёт через открытие приложения. Не контролируй AppOpen — заблокируешь каузальный путь.

---

**Инструмент Rain:**
Дождь → больше заказов, но дождь не влияет на назначение промо. Валидный инструмент для IV (Модуль 7).

---

**Вывод:**
Достаточно контролировать **Activity**, чтобы идентифицировать эффект промо.
    """)

st.divider()

# ── COLLIDER BIAS ─────────────────────────────────────────────────────────────
st.header("⚠️ Ловушка коллайдера: Berkson's Bias")
st.markdown("""
Навык и Удача независимы. Но оба влияют на найм (коллайдер).
**Среди нанятых**: низкий навык компенсируется удачей → ложная отрицательная корреляция.
""")

c5, c6 = st.columns([1, 3])
with c5:
    n_c = st.slider("Размер популяции", 200, 2000, 800)
    threshold = st.slider("Порог найма (skill + luck >)", 0.0, 2.0, 0.5, step=0.1)

rng = np.random.default_rng(99)
skill = rng.normal(0, 1, n_c)
luck  = rng.normal(0, 1, n_c)
hired = (skill + luck) > threshold
df_c  = pd.DataFrame({"skill": skill, "luck": luck,
                       "Статус": np.where(hired, "Нанят", "Не нанят")})

r_all,   _ = stats.pearsonr(df_c["skill"], df_c["luck"])
r_hired, _ = stats.pearsonr(df_c[hired]["skill"], df_c[hired]["luck"])

with c6:
    fig_c = px.scatter(
        df_c, x="skill", y="luck", color="Статус", opacity=0.55,
        color_discrete_map={"Нанят":"#4caf50","Не нанят":"#e0e0e0"},
        labels={"skill":"Навык","luck":"Удача"},
        title=f"Все: r = {r_all:.2f}  |  Только нанятые: r = {r_hired:.2f}",
    )
    st.plotly_chart(fig_c, use_container_width=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Корреляция: вся популяция", f"{r_all:.3f}")
with col_m2:
    st.metric("Корреляция: только нанятые", f"{r_hired:.3f}",
              delta="Ложная отрицательная", delta_color="inverse")

st.error("Анализ только по нанятым (survivor bias) — классическая ловушка коллайдера.")

st.divider()

# ── CODE ──────────────────────────────────────────────────────────────────────
st.header("💻 Python-код")
with st.expander("Показать код"):
    st.code("""
import networkx as nx
import matplotlib.pyplot as plt

# DAG: промо → заказы
G = nx.DiGraph()
G.add_edges_from([
    ("Activity", "Promo"),   # конфаундер
    ("Activity", "Orders"),  # конфаундер
    ("Promo",    "Orders"),  # эффект интереса
    ("Promo",    "AppOpen"), # медиатор
    ("AppOpen",  "Orders"),  # медиатор
    ("Rain",     "Orders"),  # инструмент
])

# Все пути из Promo в Orders
for path in nx.all_simple_paths(G, "Promo", "Orders"):
    print(" → ".join(path))
# Promo → Orders
# Promo → AppOpen → Orders

# Backdoor-путь (вручную): Promo ← Activity → Orders
# Для блокировки нужно контролировать Activity

# Визуализация
pos = {"Activity":(0,1), "Promo":(1,2), "AppOpen":(2,2),
       "Orders":(3,1), "Rain":(3,2.5)}
nx.draw_networkx(G, pos, with_labels=True,
                 node_size=2000, node_color="#4a90d9",
                 font_color="white", arrows=True)
plt.title("Causal DAG")
plt.axis("off")
plt.show()
""", language="python")

st.divider()
st.header("✅ Главное")
st.success("""
1. DAG — инструмент для явной записи причинно-следственных убеждений.
2. Fork (развилка) — конфаундер, создаёт ложную корреляцию. Блокируется условием.
3. Chain (цепочка) — медиатор. Условие на медиатор блокирует каузальный путь.
4. Collider (коллайдер) — условие открывает ложную связь (Berkson's bias).
5. Backdoor criterion: контролируй конфаундеры, не трогай медиаторы и потомков.
""")
render_nav(current=4)
