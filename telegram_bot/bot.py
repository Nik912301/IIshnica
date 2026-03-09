# -*- coding: utf-8 -*-
"""
Telegram-бот для сегментации растений.
Повторяет функции веб-интерфейса: загрузка изображения, предобработка (ч/б, контраст, яркость),
анализ, получение результата (картинка + статистика + JSON/CSV).
Требует запущенный бекенд (uvicorn backend.main:app).
"""

import io
import logging
import os
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# Настройки
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE = os.environ.get("PLANT_API_BASE", "http://127.0.0.1:8000")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Ключи для user_data
KEY_GRAYSCALE = "grayscale"
KEY_CONTRAST = "contrast"
KEY_BRIGHTNESS = "brightness"
KEY_SATURATION = "saturation"
KEY_BLUR = "blur"

DEFAULTS = {
    KEY_GRAYSCALE: False,
    KEY_CONTRAST: 1.0,
    KEY_BRIGHTNESS: 0,
    KEY_SATURATION: 1.0,
    KEY_BLUR: 0.0,
}


def get_user_params(context: ContextTypes.DEFAULT_TYPE) -> dict:
    user_data = context.user_data
    return {
        "grayscale": user_data.get(KEY_GRAYSCALE, DEFAULTS[KEY_GRAYSCALE]),
        "contrast": user_data.get(KEY_CONTRAST, DEFAULTS[KEY_CONTRAST]),
        "brightness": user_data.get(KEY_BRIGHTNESS, DEFAULTS[KEY_BRIGHTNESS]),
        "saturation": user_data.get(KEY_SATURATION, DEFAULTS[KEY_SATURATION]),
        "blur": user_data.get(KEY_BLUR, DEFAULTS[KEY_BLUR]),
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🌱 <b>Сегментация растений</b>\n\n"
        "Отправь мне <b>фото растения</b> (JPG/PNG) — я запущу анализ и пришлю картинку с масками, "
        "площади и длины в мм/см и файлы JSON/CSV.\n\n"
        "Команды:\n"
        "• /calibration — режим калибровки (стандартная / по фото)\n"
        "• /calibrate — калибровка по фото доски (в режиме «по фото»)\n"
        "• /settings — настройки предобработки (ч/б, контраст, яркость)\n"
        "• /help — справка",
        parse_mode="HTML",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📷 <b>Как пользоваться</b>\n\n"
        "1. Калибровка: /calibration — выбор режима (стандартная или по фото). В режиме «по фото»: /calibrate → отправь фото доски.\n"
        "2. При необходимости: /settings (ч/б, контраст, яркость).\n"
        "3. Отправь фото растения — получи картинку с масками, таблицу площадей/длин (мм и см), JSON/CSV.\n\n"
        "Параметры в настройках:\n"
        "• <b>Ч/Б</b> — черно-белое изображение\n"
        "• <b>Контраст</b> — 0.5–2.0\n"
        "• <b>Яркость</b> — от -100 до 100",
        parse_mode="HTML",
    )


def build_settings_keyboard(context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    params = get_user_params(context)
    g = params["grayscale"]
    c = params["contrast"]
    b = params["brightness"]
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"Ч/Б: {'вкл' if g else 'выкл'}",
                callback_data="toggle_grayscale",
            ),
        ],
        [
            InlineKeyboardButton("Контраст −", callback_data="contrast_down"),
            InlineKeyboardButton(f"{c:.1f}", callback_data="noop"),
            InlineKeyboardButton("Контраст +", callback_data="contrast_up"),
        ],
        [
            InlineKeyboardButton("Яркость −", callback_data="brightness_down"),
            InlineKeyboardButton(f"{b}", callback_data="noop"),
            InlineKeyboardButton("Яркость +", callback_data="brightness_up"),
        ],
        [InlineKeyboardButton("Сбросить", callback_data="settings_reset")],
    ])


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    params = get_user_params(context)
    text = (
        "⚙️ <b>Настройки предобработки</b>\n\n"
        f"• Ч/Б: {'да' if params['grayscale'] else 'нет'}\n"
        f"• Контраст: {params['contrast']:.1f}\n"
        f"• Яркость: {params['brightness']}\n"
        f"• Насыщенность: {params['saturation']:.1f}\n"
        f"• Размытие: {params['blur']:.0f}\n\n"
        "Используй кнопки ниже или отправь фото для анализа с текущими настройками."
    )
    await update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=build_settings_keyboard(context),
    )


async def calibration_mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "calib_mode_standard":
        ok = set_calibration_mode_api("standard")
        text = "✅ Режим калибровки: <b>стандартная</b>. Используется фиксированный масштаб." if ok else "❌ Не удалось переключить (проверь бекенд)."
    elif data == "calib_mode_photo":
        ok = set_calibration_mode_api("photo")
        text = (
            "✅ Режим калибровки: <b>по фото</b>.\n\n"
            "Отправь /calibrate и затем фото шахматной доски, чтобы задать масштаб по изображению."
        ) if ok else "❌ Не удалось переключить (проверь бекенд)."
    else:
        return
    try:
        await query.edit_message_text(text, parse_mode="HTML")
    except Exception:
        pass


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    if data in ("calib_mode_standard", "calib_mode_photo"):
        await calibration_mode_callback(update, context)
        return
    await query.answer()
    ud = context.user_data

    if data == "noop":
        return

    if data == "toggle_grayscale":
        ud[KEY_GRAYSCALE] = not ud.get(KEY_GRAYSCALE, DEFAULTS[KEY_GRAYSCALE])
    elif data == "contrast_down":
        ud[KEY_CONTRAST] = max(0.5, ud.get(KEY_CONTRAST, 1.0) - 0.2)
    elif data == "contrast_up":
        ud[KEY_CONTRAST] = min(2.0, ud.get(KEY_CONTRAST, 1.0) + 0.2)
    elif data == "brightness_down":
        ud[KEY_BRIGHTNESS] = max(-100, ud.get(KEY_BRIGHTNESS, 0) - 10)
    elif data == "brightness_up":
        ud[KEY_BRIGHTNESS] = min(100, ud.get(KEY_BRIGHTNESS, 0) + 10)
    elif data == "settings_reset":
        for k in (KEY_GRAYSCALE, KEY_CONTRAST, KEY_BRIGHTNESS, KEY_SATURATION, KEY_BLUR):
            ud[k] = DEFAULTS[k]

    params = get_user_params(context)
    text = (
        "⚙️ <b>Настройки предобработки</b>\n\n"
        f"• Ч/Б: {'да' if params['grayscale'] else 'нет'}\n"
        f"• Контраст: {params['contrast']:.1f}\n"
        f"• Яркость: {params['brightness']}\n"
        f"• Насыщенность: {params['saturation']:.1f}\n"
        f"• Размытие: {params['blur']:.0f}\n\n"
        "Используй кнопки ниже или отправь фото для анализа."
    )
    await query.edit_message_text(
        text,
        parse_mode="HTML",
        reply_markup=build_settings_keyboard(context),
    )


def call_analyze_api(image_bytes: bytes, filename: str, params: dict) -> dict:
    """Отправляет изображение на бекенд и возвращает ответ JSON."""
    url = f"{API_BASE.rstrip('/')}/api/analyze"
    files = {"file": (filename or "image.jpg", image_bytes, "image/jpeg")}
    data = {
        "grayscale": "true" if params["grayscale"] else "false",
        "contrast": str(params["contrast"]),
        "brightness": str(params["brightness"]),
        "saturation": str(params["saturation"]),
        "blur": str(params["blur"]),
        "conf": "0.10",
        "connect_dist": "100.0",
        "root_overlap": "0.6",
    }
    r = requests.post(url, files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def get_calibration_from_api() -> dict:
    """Текущая калибровка с бекенда (mode, mm_per_px)."""
    try:
        r = requests.get(f"{API_BASE.rstrip('/')}/api/calibration", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"mode": "standard", "mm_per_px": 0.213317502}


def set_calibration_mode_api(mode: str) -> bool:
    """Установить режим калибровки на бекенде."""
    try:
        r = requests.post(
            f"{API_BASE.rstrip('/')}/api/calibration/mode",
            json={"mode": mode},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


async def calibration_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключатель режима калибровки: стандартная / по фото."""
    data = get_calibration_from_api()
    mode = data.get("mode", "standard")
    mm = data.get("mm_per_px", 0)
    mode_label = "по фото" if mode == "photo" else "стандартная"
    text = (
        "📐 <b>Режим калибровки</b>\n\n"
        f"Сейчас: <b>{mode_label}</b>\n"
        f"Масштаб: <code>{mm:.6f}</code> мм/px\n\n"
        "• <b>Стандартная</b> — фиксированный масштаб (без фото).\n"
        "• <b>По фото</b> — масштаб по загруженной шахматной доске (команда /calibrate)."
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Стандартная", callback_data="calib_mode_standard"),
            InlineKeyboardButton("По фото", callback_data="calib_mode_photo"),
        ],
    ])
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)


async def calibrate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["await_calibration"] = True
    await update.message.reply_text(
        "📐 <b>Калибровка по фото</b>\n\n"
        "Отправь <b>фото шахматной доски</b> (клетка 10×10 мм, в кадре может быть 5×8, 7×4 или часть сетки). "
        "После этого масштаб будет применён и режим калибровки переключится на «по фото».",
        parse_mode="HTML",
    )


def call_calibrate_api(image_bytes: bytes, filename: str = "calib.jpg") -> dict:
    url = f"{API_BASE.rstrip('/')}/api/calibrate"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data = {"pattern": "5x8", "checker_size_mm": "10"}
    r = requests.post(url, files=files, data=data, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_result_file(job_id: str, kind: str) -> bytes:
    """Скачивает файл результата по job_id (image, json, csv)."""
    url = f"{API_BASE.rstrip('/')}/api/download/{kind}/{job_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN":
        await update.message.reply_text(
            "Бот не настроен: задайте переменную окружения TELEGRAM_BOT_TOKEN."
        )
        return

    # Калибровка по шахматной доске
    if context.user_data.pop("await_calibration", False):
        msg = await update.message.reply_text("⏳ Калибровка…")
        try:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            bio = io.BytesIO()
            await file.download_to_memory(bio)
            image_bytes = bio.getvalue()
        except Exception as e:
            await msg.edit_text(f"❌ Не удалось загрузить фото: {e}")
            return
        try:
            result = call_calibrate_api(image_bytes)
            mm = result.get("mm_per_px", 0)
            await msg.edit_text(
                f"✅ <b>Калибровка применена</b>\n\n"
                f"Масштаб: <code>{mm:.6f}</code> мм/px\n"
                f"Теперь все анализы будут в мм и см.",
                parse_mode="HTML",
            )
        except requests.exceptions.RequestException as e:
            await msg.edit_text(f"❌ Ошибка калибровки. Проверь фото доски (5×8, 10 мм).\n{e}")
        return

    msg = await update.message.reply_text("⏳ Обрабатываю изображение…")
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        bio = io.BytesIO()
        await file.download_to_memory(bio)
        image_bytes = bio.getvalue()
    except Exception as e:
        logger.exception("Download photo failed")
        await msg.edit_text(f"❌ Не удалось загрузить фото: {e}")
        return

    params = get_user_params(context)
    try:
        result = call_analyze_api(image_bytes, "photo.jpg", params)
    except requests.exceptions.RequestException as e:
        logger.exception("API request failed")
        await msg.edit_text(
            f"❌ Ошибка бекенда. Убедись, что сервер запущен: {API_BASE}\n\n{e}"
        )
        return
    except Exception as e:
        logger.exception("Analyze failed")
        await msg.edit_text(f"❌ Ошибка анализа: {e}")
        return

    job_id = result["job_id"]
    areas_mm2 = result.get("areas_mm2", {})
    areas_cm2 = result.get("areas_cm2", {})
    lengths_mm = result.get("lengths_mm", {})
    lengths_cm = result.get("lengths_cm", {})

    def _mm2(v):
        return f"{float(v or 0):.1f}"
    def _cm2(v):
        return f"{float(v or 0):.2f}"
    def _mm(v):
        return f"{float(v or 0):.1f}"
    def _cm(v):
        return f"{float(v or 0):.2f}"

    root_a_mm, root_a_cm = _mm2(areas_mm2.get("root")), _cm2(areas_cm2.get("root") or (areas_mm2.get("root", 0) / 100))
    stem_a_mm, stem_a_cm = _mm2(areas_mm2.get("stem")), _cm2(areas_cm2.get("stem") or (areas_mm2.get("stem", 0) / 100))
    leaf_a_mm, leaf_a_cm = _mm2(areas_mm2.get("leaf")), _cm2(areas_cm2.get("leaf") or (areas_mm2.get("leaf", 0) / 100))
    total_a_mm = _mm2(sum(areas_mm2.values()))
    total_a_cm = _cm2(sum(areas_cm2.values()) or sum(areas_mm2.values()) / 100)

    root_l_mm = _mm(lengths_mm.get("root"))
    root_l_cm = _cm(lengths_cm.get("root") or (lengths_mm.get("root", 0) / 10))
    stem_l_mm = _mm(lengths_mm.get("stem"))
    stem_l_cm = _cm(lengths_cm.get("stem") or (lengths_mm.get("stem", 0) / 10))
    leaf_l_mm = _mm(lengths_mm.get("leaf"))
    leaf_l_cm = _cm(lengths_cm.get("leaf") or (lengths_mm.get("leaf", 0) / 10))
    total_l_mm = _mm(sum(lengths_mm.values()))
    total_l_cm = _cm(sum(lengths_cm.values()) or sum(lengths_mm.values()) / 10)

    caption = (
        "🌿 <b>Результат анализа</b>\n\n"
        "▫️ <b>Длина</b> (мм → см)\n"
        f"   Корень:   {root_l_mm} → <b>{root_l_cm}</b>\n"
        f"   Стебель:  {stem_l_mm} → <b>{stem_l_cm}</b>\n"
        f"   Листья:   {leaf_l_mm} → <b>{leaf_l_cm}</b>\n"
        f"   ─────────────\n"
        f"   Всего:    {total_l_mm} → <b>{total_l_cm}</b>\n\n"
        "▫️ <b>Площадь</b> (мм² → см²)\n"
        f"   Корень:   {root_a_mm} → <b>{root_a_cm}</b>\n"
        f"   Стебель:  {stem_a_mm} → <b>{stem_a_cm}</b>\n"
        f"   Листья:   {leaf_a_mm} → <b>{leaf_a_cm}</b>\n"
        f"   ─────────────\n"
        f"   Всего:    {total_a_mm} → <b>{total_a_cm}</b>\n\n"
        f"<code>id: {job_id[:12]}…</code>"
    )

    # Скачиваем картинку результата и отправляем
    try:
        img_data = fetch_result_file(job_id, "image")
        await msg.delete()
        await update.message.reply_photo(
            photo=io.BytesIO(img_data),
            caption=caption,
            parse_mode="HTML",
        )
    except Exception as e:
        logger.exception("Send result image failed")
        await msg.edit_text(caption + f"\n\n⚠ Не удалось отправить картинку: {e}")
        return

    # Отправляем JSON и CSV как документы
    try:
        json_data = fetch_result_file(job_id, "json")
        await update.message.reply_document(
            document=io.BytesIO(json_data),
            filename=f"{job_id}_results.json",
            caption="Результаты в JSON",
        )
    except Exception as e:
        logger.warning("Send JSON failed: %s", e)

    try:
        csv_data = fetch_result_file(job_id, "csv")
        await update.message.reply_document(
            document=io.BytesIO(csv_data),
            filename=f"{job_id}_results.csv",
            caption="Результаты в CSV",
        )
    except Exception as e:
        logger.warning("Send CSV failed: %s", e)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    from telegram.error import Conflict
    if context.error and isinstance(context.error, Conflict):
        logger.error(
            "Конфликт: уже запущен другой экземпляр бота с этим токеном. "
            "Закрой все остальные окна/процессы с ботом и запусти только один."
        )
        import sys
        sys.exit(1)
    logger.exception("Exception while handling an update: %s", context.error)


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN":
        print("Задайте TELEGRAM_BOT_TOKEN (токен от @BotFather).")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("calibration", calibration_cmd))
    app.add_handler(CommandHandler("calibrate", calibrate_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CallbackQueryHandler(settings_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Бот запущен. Бекенд: %s", API_BASE)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
