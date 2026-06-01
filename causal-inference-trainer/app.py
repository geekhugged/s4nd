import streamlit as st

st.set_page_config(
    page_title="Causal Inference Trainer",
    page_icon="🔬",
    layout="wide",
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔬 Causal Inference Trainer")
st.markdown(
    "Пошаговый тренажёр по причинно-следственному анализу. "
    "Каждый модуль: **теория** → **интерактивная симуляция** → **Python-код** → **пример из food delivery**."
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍 Программа")
    st.caption("Выбери модуль в меню выше ↑")
    st.divider()
    st.caption("Все примеры — из food delivery")

# ── MODULE CARDS ──────────────────────────────────────────────────────────────
MODULES = [
    (1, "🔗", "Корреляция vs Причинность",
     "Конфаундеры, ложные корреляции, первый вопрос причинности.", True),
    (2, "🎯", "Potential Outcomes",
     "Модель Рубина, контрфактуалы, ATE, selection bias.", True),
    (3, "📊", "DAGs",
     "Причинные графы, fork/chain/collider, backdoor criterion.", True),
    (4, "🧪", "A/B Testing",
     "RCT, рандомизация, статистическая мощность.", False),
    (5, "⚖️", "Matching & IPW",
     "Propensity score, matching, inverse probability weighting.", False),
    (6, "📈", "Difference-in-Differences",
     "DiD, parallel trends, event study.", False),
    (7, "🎲", "Instrumental Variables",
     "IV, 2SLS, как искать инструменты.", False),
    (8, "📉", "Regression Discontinuity",
     "RDD, sharp vs fuzzy, выбор bandwidth.", False),
    (9, "🤖", "Causal ML",
     "Double ML, CATE, heterogeneous effects, EconML.", False),
]

st.subheader("📍 Программа")
cols = st.columns(3)
for i, (num, emoji, title, desc, ready) in enumerate(MODULES):
    with cols[i % 3]:
        bg     = "#e8f5e9" if ready else "#f5f5f5"
        border = "#4caf50" if ready else "#e0e0e0"
        badge  = "✅ Открыт" if ready else "🔜 Скоро"
        color  = "#2e7d32" if ready else "#9e9e9e"
        st.markdown(f"""
<div style="border:1px solid {border};border-radius:10px;
            padding:14px;margin:6px 0;background:{bg};min-height:130px">
  <div style="font-weight:600;font-size:15px">{emoji} {num}. {title}</div>
  <div style="color:#555;font-size:13px;margin:6px 0 10px">{desc}</div>
  <span style="color:{color};font-size:12px;font-weight:600">{badge}</span>
</div>""", unsafe_allow_html=True)

# ── WHY CAUSAL INFERENCE ──────────────────────────────────────────────────────
st.divider()
st.subheader("🍕 Почему это важно в food delivery?")
c1, c2, c3 = st.columns(3)
with c1:
    st.info("**Промоакции**\n\nПромо увеличило заказы — или эти пользователи и так бы заказали?")
with c2:
    st.info("**Ценообразование**\n\nСнижение цены подняло спрос — или просто наступил сезон?")
with c3:
    st.info("**Операции**\n\nБыстрая доставка увеличивает retention — или retention растёт по другим причинам?")

st.divider()
st.subheader("🗺️ Путь обучения")
st.markdown("""
```
Корреляция vs Причинность          ← начни здесь
        ↓
Potential Outcomes (ATE, selection bias)
        ↓
DAGs (backdoor criterion)
        ↓
A/B Testing  →  Matching  →  DiD  →  IV  →  RDD
                                              ↓
                                         Causal ML
```
Каждый следующий модуль строится на предыдущем.
""")
st.caption("Используй боковое меню слева для навигации между модулями.")
