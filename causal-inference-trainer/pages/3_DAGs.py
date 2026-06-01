import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="DAGs", page_icon="📊", layout="wide")

st.title("📊 Модуль 3: DAGs — Причинные Графы")

# ── THEORY ────────────────────────────────────────────────────────────────────
st.markdown("""
## Теория

### Что такое DAG?

**DAG** (Directed Acyclic Graph) — ориентированный граф без циклов, где:
- **Узлы** = переменные
- **Стрелки** = прямые причинные связи ($X \\rightarrow Y$ означает «$X$ причиняет $Y$»)
- **Нет циклов** — причина не может быть следствием самой себя

DAG — это **декларация наших убеждений** о причинно-следственной структуре мира.

### Три фундаментальные структуры

| Структура | Граф | Свойство |
|-----------|------|----------|
| **Цепочка (Chain)** | $X \\rightarrow Z \\rightarrow Y$ | $Z$ блокирует путь при условии на $Z$ |
| **Развилка (Fork)** | $X \\leftarrow Z \\rightarrow Y$ | $Z$ — конфаундер; блокируется при условии на $Z$ |
| **Коллайдер (Collider)** | $X \\rightarrow Z \\leftarrow Y$ | Путь заблокирован; **открывается** при условии на $Z$ |

### Backdoor Criterion

Чтобы идентифицировать каузальный эффект $X \\rightarrow Y$, нужно **заблокировать все backdoor-пути** (пути из $X$ в $Y$ идущие «против» стрелки из $X$).

Набор переменных $Z$ удовлетворяет backdoor criterion если:
1. $Z$ блокирует все backdoor-пути между $X$ и $Y$
2. $Z$ не содержит потомков $X$

### Правило d-separation

Переменные $X$ и $Y$ **d-разделены** (условно независимы) при условии $Z$ если $Z$ блокирует все пути:
- **Цепочки и развилки** блокируются при условии на переменную в середине пути
- **Коллайдеры** открываются при условии на коллайдер или его потомка
""")

st.divider()

# ── GRAPH STRUCTURES VISUALIZATION ───────────────────────────────────────────
st.header("🔍 Три фундаментальные структуры")


def draw_dag(edges, node_labels=None, highlight_nodes=None, title="", figsize=(4, 3)):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=2)
    colors = []
    for node in G.nodes():
        if highlight_nodes and node in highlight_nodes:
            colors.append("#ff6b35")
        else:
            colors.append("#4a90d9")
    nx.draw(G, pos, ax=ax, with_labels=True,
            labels=node_labels or {n: n for n in G.nodes()},
            node_color=colors, node_size=1800,
            font_size=11, font_color="white", font_weight="bold",
            arrows=True, arrowsize=20,
            edge_color="#555", width=2,
            connectionstyle="arc3,rad=0.1")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    return fig


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Цепочка (Chain)")
    fig1 = draw_dag(
        [("X", "Z"), ("Z", "Y")],
        title="X → Z → Y",
        highlight_nodes=["Z"],
    )
    st.pyplot(fig1)
    st.markdown("""
    **X** влияет на **Y** через медиатор **Z**.

    - Условие на **Z** блокирует путь
    - *X ⊥ Y | Z*

    Пример: Промо → Первый заказ → Retention
    """)

with col2:
    st.subheader("Развилка (Fork)")
    fig2 = draw_dag(
        [("Z", "X"), ("Z", "Y")],
        title="X ← Z → Y",
        highlight_nodes=["Z"],
    )
    st.pyplot(fig2)
    st.markdown("""
    **Z** — конфаундер, создаёт ложную корреляцию X~Y.

    - Условие на **Z** блокирует путь
    - *X ⊥ Y | Z*

    Пример: Сезон → Курьеры, Сезон → Заказы
    """)

with col3:
    st.subheader("Коллайдер (Collider)")
    fig3 = draw_dag(
        [("X", "Z"), ("Y", "Z")],
        title="X → Z ← Y",
        highlight_nodes=["Z"],
    )
    st.pyplot(fig3)
    st.markdown("""
    **Z** — коллайдер, **блокирует** путь по умолчанию.

    - Условие на **Z** **открывает** путь (опасно!)
    - Без условия: *X ⊥ Y*

    Пример: Способность + Удача → Нанят
    """)

plt.close("all")

st.divider()

# ── FOOD DELIVERY DAG ─────────────────────────────────────────────────────────
st.header("🍕 DAG для food delivery: эффект промо на заказы")

st.markdown("""
Строим причинный граф для типичной задачи: **влияет ли промо на количество заказов?**

Переменные:
- **Promo** — получил ли пользователь промокод
- **Orders** — число заказов
- **Activity** — историческая активность пользователя (конфаундер)
- **AppOpen** — открытия приложения (медиатор)
- **Rain** — дождь (инструмент — влияет на заказы, но не на промо)
""")

col_dag, col_explain = st.columns([2, 1])

with col_dag:
    G_delivery = nx.DiGraph()
    edges_delivery = [
        ("Activity", "Promo"),
        ("Activity", "Orders"),
        ("Promo", "AppOpen"),
        ("AppOpen", "Orders"),
        ("Promo", "Orders"),
        ("Rain", "Orders"),
    ]
    G_delivery.add_edges_from(edges_delivery)

    pos_delivery = {
        "Activity": (0, 1),
        "Promo": (1, 2),
        "AppOpen": (2, 2),
        "Orders": (3, 1),
        "Rain": (3, 2.5),
    }

    node_colors = {
        "Activity": "#ff6b35",
        "Promo": "#4a90d9",
        "AppOpen": "#7bc67e",
        "Orders": "#4a90d9",
        "Rain": "#9b59b6",
    }

    fig_del, ax_del = plt.subplots(figsize=(8, 4))
    colors = [node_colors[n] for n in G_delivery.nodes()]
    nx.draw(G_delivery, pos_delivery, ax=ax_del,
            with_labels=True,
            node_color=colors, node_size=2200,
            font_size=10, font_color="white", font_weight="bold",
            arrows=True, arrowsize=20,
            edge_color="#555", width=2,
            connectionstyle="arc3,rad=0.05")

    legend_elements = [
        mpatches.Patch(color="#ff6b35", label="Конфаундер"),
        mpatches.Patch(color="#4a90d9", label="Основные переменные"),
        mpatches.Patch(color="#7bc67e", label="Медиатор"),
        mpatches.Patch(color="#9b59b6", label="Инструмент (IV)"),
    ]
    ax_del.legend(handles=legend_elements, loc="lower left", fontsize=9)
    ax_del.set_title("Causal DAG: Promo → Orders", fontsize=13, fontweight="bold")
    fig_del.patch.set_facecolor("#fafafa")
    ax_del.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig_del)
    plt.close("all")

with col_explain:
    st.markdown("""
    **Backdoor-пути из Promo в Orders:**

    `Promo ← Activity → Orders`

    Это единственный backdoor-путь.
    Чтобы идентифицировать эффект промо — нужно **контролировать Activity**.

    ---

    **Медиатор AppOpen:**

    Часть эффекта промо идёт через открытие приложения.
    Если контролировать AppOpen — мы заблокируем часть каузального пути.

    ---

    **Инструмент Rain:**

    Дождь влияет на заказы, но **не** на назначение промо.
    Это валидный инструмент для IV-подхода (Модуль 7).
    """)

st.divider()

# ── COLLIDER BIAS SIMULATION ──────────────────────────────────────────────────
st.header("⚠️ Симуляция: Ловушка коллайдера")
st.markdown("""
**Пример:** Компания анализирует данные только по **нанятым** курьерам.

- Навык (Skill) и Удача (Luck) независимы в популяции
- Но оба влияют на найм (Hired) — коллайдер
- Среди нанятых: низкий навык компенсируется высокой удачей и наоборот → **ложная отрицательная корреляция**
""")

col_ctrl_c, col_plot_c = st.columns([1, 3])

with col_ctrl_c:
    n_c = st.slider("Размер популяции", 200, 2000, 800, key="coll_n")
    threshold = st.slider("Порог найма (skill + luck >)", 0.0, 2.0, 0.5, step=0.1)

rng = np.random.default_rng(99)
skill = rng.normal(0, 1, n_c)
luck = rng.normal(0, 1, n_c)
hired = (skill + luck) > threshold
df_c = pd.DataFrame({"skill": skill, "luck": luck, "hired": hired})

r_all, _ = stats.pearsonr(df_c["skill"], df_c["luck"])
r_hired, _ = stats.pearsonr(df_c[df_c.hired]["skill"], df_c[df_c.hired]["luck"])

with col_plot_c:
    fig_c = px.scatter(
        df_c, x="skill", y="luck",
        color=df_c["hired"].map({True: "Нанят", False: "Не нанят"}),
        color_discrete_map={"Нанят": "#4caf50", "Не нанят": "#e0e0e0"},
        opacity=0.6,
        labels={"skill": "Навык", "luck": "Удача", "color": ""},
        title=f"Все: r = {r_all:.2f}  |  Только нанятые: r = {r_hired:.2f}",
    )
    st.plotly_chart(fig_c, use_container_width=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Корреляция в полной выборке", f"{r_all:.3f}", help="Навык и Удача независимы")
with col_m2:
    st.metric("Корреляция среди нанятых", f"{r_hired:.3f}",
              delta="Ложная отрицательная", delta_color="inverse")

st.error(
    "Если анализировать только нанятых курьеров — навык и удача кажутся **отрицательно** связанными. "
    "Это Berkson's bias — типичная ловушка коллайдера."
)

st.divider()

# ── CODE ──────────────────────────────────────────────────────────────────────
st.header("💻 Python: строим DAG с networkx")

with st.expander("Показать код"):
    st.code("""
import networkx as nx
import matplotlib.pyplot as plt

# Строим DAG для задачи промо → заказы
G = nx.DiGraph()
G.add_edges_from([
    ("Activity", "Promo"),    # конфаундер
    ("Activity", "Orders"),   # конфаундер
    ("Promo", "Orders"),      # основной эффект
    ("Promo", "AppOpen"),     # медиатор
    ("AppOpen", "Orders"),    # медиатор
    ("Rain", "Orders"),       # инструмент
])

# Находим все пути из Promo в Orders
all_paths = list(nx.all_simple_paths(G, "Promo", "Orders"))
print("Все прямые пути:", all_paths)

# Backdoor-пути (идут против стрелки из Promo)
# Ищем вручную: Promo <- Activity -> Orders
# Для автоматической идентификации используйте dowhy или causaldag

# Визуализация
pos = {"Activity": (0,1), "Promo": (1,2), "AppOpen": (2,2),
       "Orders": (3,1), "Rain": (3,2.5)}
nx.draw(G, pos, with_labels=True, node_size=2000,
        node_color="#4a90d9", font_color="white",
        font_weight="bold", arrows=True, arrowsize=20)
plt.title("Causal DAG: Promo → Orders")
plt.show()
""", language="python")

st.divider()

# ── TAKEAWAYS ─────────────────────────────────────────────────────────────────
st.header("✅ Главное из модуля")
st.success("""
1. **DAG** — инструмент для записи наших предположений о причинно-следственных связях.
2. **Три структуры**: цепочка (медиатор), развилка (конфаундер), коллайдер.
3. **Backdoor criterion**: чтобы найти истинный эффект X→Y, блокируй все backdoor-пути.
4. **Коллайдер — ловушка**: условие на коллайдер *открывает* ложную связь.
5. В food delivery: Activity — конфаундер для оценки эффекта промо. AppOpen — медиатор. Rain — потенциальный инструмент.
""")
st.markdown("**👉 Следующий модуль:** A/B Testing (скоро)")
