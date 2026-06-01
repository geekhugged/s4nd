# 🔬 Causal Inference Trainer

Интерактивный тренажёр по причинно-следственному анализу.
Каждый модуль: **теория** → **симуляция** → **Python-код** → **пример из food delivery**.

## Запуск

```bash
cd causal-inference-trainer
pip install -r requirements.txt
streamlit run app.py
```

## Модули

| # | Тема | Статус |
|---|------|--------|
| 1 | 🔗 Корреляция vs Причинность | ✅ |
| 2 | 🎯 Potential Outcomes | ✅ |
| 3 | 📊 DAGs | ✅ |
| 4 | 🧪 A/B Testing | 🔜 |
| 5 | ⚖️ Matching & IPW | 🔜 |
| 6 | 📈 Difference-in-Differences | 🔜 |
| 7 | 🎲 Instrumental Variables | 🔜 |
| 8 | 📉 Regression Discontinuity | 🔜 |
| 9 | 🤖 Causal ML | 🔜 |

## Структура

```
causal-inference-trainer/
├── app.py                  # Главная страница
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Тема
├── utils/
│   ├── generators.py       # Генераторы данных
│   └── ui.py               # Общие компоненты (sidebar, прогресс, навигация)
└── pages/
    ├── 1_Correlation.py
    ├── 2_Potential_Outcomes.py
    └── 3_DAGs.py
```
