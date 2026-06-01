import streamlit as st

MODULES = [
    (1, "🔗", "Корреляция vs Причинность", True),
    (2, "🎯", "Potential Outcomes",          True),
    (3, "📊", "DAGs",                         True),
    (4, "🧪", "A/B Testing",                  False),
    (5, "⚖️", "Matching & IPW",               False),
    (6, "📈", "Difference-in-Differences",    False),
    (7, "🎲", "Instrumental Variables",       False),
    (8, "📉", "Regression Discontinuity",     False),
    (9, "🤖", "Causal ML",                    False),
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
    completed = current - 1
    available = sum(1 for _, _, _, r in MODULES if r)
    dots = ""
    for num, emoji, _, ready in MODULES:
        if num < current:
            dots += f"✅ "
        elif num == current:
            dots += f"**{emoji}** "
        elif ready:
            dots += f"○ "
        else:
            dots += f"· "
    st.caption(f"Модуль {current} / {len(MODULES)} &nbsp;|&nbsp; Доступно: {available}  &nbsp;|&nbsp; {dots}")
    st.progress(current / len(MODULES))


def render_nav(current: int) -> None:
    prev_available = next(
        ((num, title) for num, _, title, ready in reversed(MODULES[:current-1]) if ready), None
    )
    next_available = next(
        ((num, title) for num, _, title, ready in MODULES[current:] if ready), None
    )

    st.divider()
    col_prev, col_mid, col_next = st.columns([2, 6, 2])
    with col_prev:
        if prev_available:
            st.markdown(f"← Модуль {prev_available[0]}\n\n*{prev_available[1]}*")
    with col_next:
        if next_available:
            st.markdown(f"Модуль {next_available[0]} →\n\n*{next_available[1]}*",
                        help="Используй боковое меню для перехода")
        else:
            st.markdown("*Скоро: следующий модуль*")
