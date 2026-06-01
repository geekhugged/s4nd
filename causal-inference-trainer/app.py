import streamlit as st

st.set_page_config(
    page_title="Causal Inference Trainer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Causal Inference Trainer")
st.markdown(
    "Пошаговый тренажёр по причинно-следственному анализу — "
    "от базовых концепций до Causal ML. "
    "Каждая тема: **теория** + **интерактивные симуляции** + **примеры из food delivery**."
)

st.divider()

MODULES = [
    (1, "🔗", "Корреляция vs Причинность",
     "Конфаундеры, ложные корреляции, первый вопрос причинности.", True),
    (2, "🎯", "Potential Outcomes",
     "Модель Рубина, контрфактуалы, ATE, selection bias.", True),
    (3, "📊", "DAGs",
     "Причинные графы, backdoor criterion, d-separation.", True),
    (4, "🧪", "A/B тестирование",
     "RCT, рандомизация, статистическая мощность.", False),
    (5, "⚖️", "Matching & IPW",
     "Propensity score, matching, inverse probability weighting.", False),
    (6, "📈", "Difference-in-Differences",
     "DiD, parallel trends assumption, event study.", False),
    (7, "🎲", "Instrumental Variables",
     "IV, 2SLS, поиск инструментов.", False),
    (8, "📉", "Regression Discontinuity",
     "RDD, sharp vs fuzzy, выбор bandwidth.", False),
    (9, "🤖", "Causal ML",
     "Double ML, CATE, heterogeneous effects, EconML.", False),
]

st.subheader("📍 Программа")
cols = st.columns(3)
for i, (num, emoji, title, desc, ready) in enumerate(MODULES):
    with cols[i % 3]:
        bg = "#e8f5e9" if ready else "#f5f5f5"
        border = "#4caf50" if ready else "#e0e0e0"
        badge = "✅ Доступен" if ready else "🔜 Скоро"
        badge_color = "#2e7d32" if ready else "#9e9e9e"
        st.markdown(f"""
<div style="border:1px solid {border};border-radius:10px;
            padding:14px;margin:6px 0;background:{bg}">
  <div style="font-weight:600;font-size:16px">{emoji} {num}. {title}</div>
  <div style="color:#555;font-size:13px;margin:5px 0 8px">{desc}</div>
  <span style="color:{badge_color};font-size:12px;font-weight:500">{badge}</span>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("🍕 Почему causal inference важен в food delivery?")
c1, c2, c3 = st.columns(3)
with c1:
    st.info("**Промоакции**\n\nПромо увеличило заказы — или эти пользователи и так бы заказали?")
with c2:
    st.info("**Ценообразование**\n\nСнижение цены подняло спрос — или просто наступил сезон?")
with c3:
    st.info("**Операции**\n\nБыстрая доставка увеличивает retention — или retention растёт по другим причинам?")

st.markdown("---")
st.caption("Начни с первого модуля → используй боковое меню слева.")
