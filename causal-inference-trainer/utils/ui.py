import streamlit as st

MODULES = [
    (1,  "📐", "Базовая статистика",             True),
    (2,  "🔗", "Корреляция vs Причинность",       True),
    (3,  "🎯", "Potential Outcomes",               True),
    (4,  "📊", "DAGs",                             True),
    (5,  "🧪", "A/B Testing",                      False),
    (6,  "⚖️", "Matching & IPW",                   False),
    (7,  "📈", "Difference-in-Differences",        False),
    (8,  "🎲", "Instrumental Variables",           False),
    (9,  "📉", "Regression Discontinuity",         False),
    (10, "🤖", "Causal ML",                        False),
]


def render_sidebar(current: int) -> None:
    with st.sidebar:
        st.markdown("### 📍 Программа")
        for num, emoji, title, ready in MODULES:
            if num < current:
                st.markdown(f"✅ {emoji} {num}. {title}")
            elif num == current:
                st.markdown(f"**▶ {emoji} {num}. {title}**")
            elif ready:
                st.markdown(f"○ {emoji} {num}. {title}")
            else:
                st.markdown(
                    f"<span style='color:#bbb'>🔜 {emoji} {num}. {title}</span>",
                    unsafe_allow_html=True,
                )
        st.divider()
        st.caption("Все примеры — из food delivery")


def render_progress(current: int) -> None:
    available = sum(1 for *_, r in MODULES if r)
    dots = ""
    for num, emoji, _, ready in MODULES:
        if num < current:
            dots += "✅ "
        elif num == current:
            dots += f"**{emoji}** "
        elif ready:
            dots += "○ "
        else:
            dots += "· "
    st.caption(
        f"Модуль {current} / {len(MODULES)} &nbsp;|&nbsp; "
        f"Доступно: {available} &nbsp;|&nbsp; {dots}"
    )
    st.progress(current / len(MODULES))


def render_nav(current: int) -> None:
    prev_ok = next(
        ((n, t) for n, _, t, r in reversed(MODULES[:current - 1]) if r), None
    )
    next_ok = next(
        ((n, t) for n, _, t, r in MODULES[current:] if r), None
    )
    st.divider()
    col_prev, _, col_next = st.columns([2, 6, 2])
    with col_prev:
        if prev_ok:
            st.markdown(f"← Модуль {prev_ok[0]}\n\n*{prev_ok[1]}*")
    with col_next:
        if next_ok:
            st.markdown(
                f"Модуль {next_ok[0]} →\n\n*{next_ok[1]}*",
                help="Используй боковое меню для перехода",
            )
        else:
            st.markdown("*Скоро: следующий модуль*")
