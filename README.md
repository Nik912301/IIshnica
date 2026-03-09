# Сегментация растений — веб, API и Telegram-бот

Веб-интерфейс и бот для анализа изображений растений: сегментация (корень, стебель, лист), площади и длины в мм/см. Используется стандартная калибровка (фиксированный масштаб).

---

## Что нужно для запуска

- **Python 3.10+**
- **PyTorch** (желательно с CUDA для скорости)
- **Веса модели YOLO** — файл `best.pt` (см. раздел ниже)

---

## Веса модели YOLO (обязательно)

В репозитории **нет файла весов** — он слишком большой для GitHub (>100 MB). Без весов бекенд не запустится.

**Куда положить файл:**

```
all_data/yolo_weights/plants_optimized_seg/weights/best.pt
```

Создай папки, если их нет: `all_data/yolo_weights/plants_optimized_seg/weights/` и помести туда `best.pt`.

**Откуда взять веса:**

| Вариант | Описание |
|--------|-----------|
| **У автора репозитория** | Напиши владельцу проекта (например, через Issues на GitHub) — он может выложить веса на Google Drive / Яндекс.Диск / другой файлообменник. |
| **Обучить свою модель** | В проекте есть разметка в `all_data/` (train/val, `custom_data.yaml`). Можно дообучить YOLO-seg по инструкции Ultralytics и сохранить веса как `best.pt` в указанную папку. |
| **Своя сегментационная модель** | Если у тебя уже есть свои веса YOLO-seg (корень/стебель/лист), положи их как `best.pt` в ту же папку. |

После того как `best.pt` окажется в нужном месте, запускай бекенд — всё будет работать.

---

## Установка (один раз)

### 1. Перейди в папку проекта

```bash
cd путь/к/проекту/proj
```

Например в PowerShell (Windows):

```powershell
cd "c:\Users\leonid\Downloads\проект\proj"
```

### 2. Создай виртуальное окружение (рекомендуется)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

Если PyTorch ещё не установлен:

```bash
pip install torch torchvision
```

(На Windows с NVIDIA GPU можно взять вариант с CUDA с сайта PyTorch.)

---

## Запуск

### Шаг 1: Запуск бекенда (обязательно первым)

Из корня проекта (`proj`):

**Windows (PowerShell):**

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Linux / macOS:**

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Проверка: открой в браузере [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — должен быть ответ `{"status":"ok"}`.

### Шаг 2: Веб-интерфейс

Открой в браузере:

**http://127.0.0.1:8000**

Либо открой файл `frontend/index.html` (двойной клик) — страница обращается к бекенду по адресу выше.

Дальше: загрузка фото растения → при необходимости настройки предобработки → «Запустить анализ» → просмотр результата и таблицы.

### Шаг 3: Telegram-бот (по желанию)

1. Создай бота в Telegram через [@BotFather](https://t.me/BotFather) и скопируй токен.
2. Запусти бота (бекенд должен быть уже запущен).

**Windows (PowerShell):**

```powershell
$env:TELEGRAM_BOT_TOKEN = "твой_токен_от_BotFather"
python -m telegram_bot.bot
```

**Linux / macOS:**

```bash
export TELEGRAM_BOT_TOKEN="твой_токен_от_BotFather"
python -m telegram_bot.bot
```

В боте: отправь фото растения для анализа; команда `/settings` — настройки предобработки (ч/б, контраст, яркость).

---

## Калибровка (масштаб в мм/см)

Используется **только стандартная калибровка** — фиксированный масштаб. Значение задаётся переменной окружения `MM_PER_PX` (по умолчанию подобрано под типовую съёмку). Результаты выдаются в мм²/см² и мм/см.

---

## Структура проекта

```
proj/
├── backend/
│   └── main.py          # FastAPI: анализ, калибровка, раздачи файлов
├── frontend/
│   └── index.html       # Веб-интерфейс
├── telegram_bot/
│   └── bot.py           # Telegram-бот
├── all_data/
│   └── yolo_weights/plants_optimized_seg/weights/
│       └── best.pt     # Веса YOLO (в репо нет — см. раздел «Веса модели YOLO»)
├── runs/                # Результаты анализов (создаётся при первом запуске)
├── calibration.json     # Не используется (оставлен для совместимости)
├── requirements.txt
└── README.md
```

---

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather (обязателен для бота). |
| `PLANT_API_BASE` | Адрес бекенда (по умолчанию `http://127.0.0.1:8000`). |
| `MM_PER_PX` | Масштаб «стандартной» калибровки (мм на пиксель). |

---

## Быстрый старт (копируй и подставляй пути/токен)

**Windows PowerShell:**

```powershell
cd "c:\Users\leonid\Downloads\проект\proj"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Терминал 1 — бекенд:
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
# Терминал 2 — бот (опционально):
$env:TELEGRAM_BOT_TOKEN = "ТВОЙ_ТОКЕН"
python -m telegram_bot.bot
```

Открой в браузере: **http://127.0.0.1:8000**
