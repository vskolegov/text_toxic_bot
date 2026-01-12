import os
import json
import tempfile
import subprocess
from pathlib import Path
from functools import wraps
import asyncio
import urllib.request
import urllib.error

from telegram import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

from .inference import predict_toxicity
from dotenv import load_dotenv
import whisper

load_dotenv()

BASE = Path(__file__).parent
USERS_FILE = BASE / 'data' / 'users.json'
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
if not USERS_FILE.exists():
    USERS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2))

DEFAULT_MODEL = 'voting'
DEFAULT_SHOW_TRANSCRIPT = True
DEFAULT_MODE_SELECTION = 'persistent'

# Whisper model will be loaded inside executor (blocking)
_whisper_model = None

def _read_users():
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}

def _write_users(data):
    USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def get_models_info():
    info_path = BASE / 'models_info.json'
    if info_path.exists():
        try:
            return json.loads(info_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    # fallback description
    return {
        "logreg": "LogReg — логистическая регрессия на TF-IDF.",
        "svm": "SVM — опорные векторы на TF-IDF.",
        "Ансамбль": "Ансамбль — голосование классических моделей (logreg+svm).",
        "bilstm": "BiLSTM — рекуррентная модель на PyTorch."
    }

def get_user(user_id):
    data = _read_users()
    return data.get(str(user_id), {
        'model': DEFAULT_MODEL,
        'show_transcript': DEFAULT_SHOW_TRANSCRIPT,
        'mode_selection': DEFAULT_MODE_SELECTION
    })

def set_user(user_id, **kwargs):
    data = _read_users()
    u = data.get(str(user_id), {})
    u.update(kwargs)
    data[str(user_id)] = u
    _write_users(data)

WELCOME = (
    'Привет! Я VoxToxic — бот для проверки токсичности голосовых сообщений.\n'
    'Отправь голосовое сообщение, чтобы получить транскрибацию и результат модели.\n\n'
    'Команды:\n'
    '/model — выбрать модель\n'
    '/mode — переключить режим выбора модели (постоянный/по запросу)\n'
    '/transcript — показать/скрыть транскрибацию'
)

def build_main_keyboard():
    kb = [[KeyboardButton('Выбрать модель'), KeyboardButton('Инфо моделей')],
          [KeyboardButton('Помощь')]]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

# Blocking helpers (run in executor)
def _convert_to_wav_blocking(in_path: str, out_path: str):
    cmd = ['ffmpeg', '-y', '-i', str(in_path), '-ar', '16000', '-ac', '1', str(out_path)]
    subprocess.run(cmd, check=True)

def _load_whisper_blocking(name='small'):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(name)
    return _whisper_model

def _transcribe_and_predict_blocking(wav_path: str, model_name: str):
    # wav_path is string path to wav
    model = _load_whisper_blocking('small')
    result = model.transcribe(str(wav_path), language='ru')
    text = result.get('text', '')
    res = predict_toxicity(text, model_name)
    return text, res

def label_to_text(label):
    try:
        return 'токсично' if int(label) == 1 else 'не токсично'
    except Exception:
        return str(label)

# Handlers (async)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(f'Привет, {user.first_name}!')
    await update.message.reply_text(WELCOME, reply_markup=build_main_keyboard())

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)

async def choose_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = [InlineKeyboardButton('LogReg', callback_data='model:logreg'),
               InlineKeyboardButton('SVM', callback_data='model:svm'),
               InlineKeyboardButton('Ансамбль', callback_data='model:voting'),
               InlineKeyboardButton('BiLSTM', callback_data='model:bilstm')]
    kb = InlineKeyboardMarkup([[b] for b in buttons])
    await update.message.reply_text('Выберите модель:', reply_markup=kb)

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    u = get_user(user.id)
    current = u.get('mode_selection', DEFAULT_MODE_SELECTION)
    new = 'per_request' if current == 'persistent' else 'persistent'
    set_user(user.id, mode_selection=new)
    await update.message.reply_text(f'Режим выбора модели: {new}')

async def transcript_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    u = get_user(user.id)
    current = u.get('show_transcript', DEFAULT_SHOW_TRANSCRIPT)
    new = not current
    set_user(user.id, show_transcript=new)
    await update.message.reply_text(f'Показывать транскрибацию: {new}')

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip().lower()
    if txt == 'выбрать модель':
        return await choose_model_cmd(update, context)
    if txt == 'инфо моделей' or txt == 'инфо':
        info = get_models_info()
        lines = [f"{key}: {val}" for key, val in info.items()]
        await_or_reply = getattr(update.message, 'reply_text', update.message.reply_text)
        # synchronous handler may be async; use reply_text directly (function is async in v20)
        await update.message.reply_text("\n".join(lines))
        return
    if txt == 'помощь':
        return await help_cmd(update, context)
    await update.message.reply_text('Неизвестная команда. Используйте меню.')


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    u = get_user(user.id)

    voice = update.message.voice or update.message.audio
    if voice is None:
        await update.message.reply_text('Пожалуйста пришлите голосовое сообщение.')
        return

    file_id = voice.file_id
    f = await context.bot.get_file(file_id)
    tmp_in = Path(tempfile.mktemp(suffix='.ogg'))
    tmp_wav = Path(tempfile.mktemp(suffix='.wav'))
    await f.download_to_drive(str(tmp_in))

    loop = asyncio.get_running_loop()
    try:
        # convert
        await loop.run_in_executor(None, _convert_to_wav_blocking, str(tmp_in), str(tmp_wav))
        # transcribe + predict (blocking, in executor)
        model_choice = u.get('model', DEFAULT_MODEL)
        text, res = await loop.run_in_executor(None, _transcribe_and_predict_blocking, str(tmp_wav), model_choice)
    except Exception as e:
        await update.message.reply_text('Ошибка обработки аудио.')
        tmp_in.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)
        return

    if u.get('show_transcript', DEFAULT_SHOW_TRANSCRIPT):
        await update.message.reply_text(f'Транскрибация:\n"{text}"')

    mode = u.get('mode_selection', DEFAULT_MODE_SELECTION)
    if mode == 'persistent':
        await update.message.reply_text(f'Результат: {label_to_text(res["label"])} — {res["prob"]*100:.1f}% (модель: {res["model"]})')
    else:
        buttons = [InlineKeyboardButton('LogReg', callback_data=f'run:logreg:{text}'),
                   InlineKeyboardButton('SVM', callback_data=f'run:svm:{text}'),
                   InlineKeyboardButton('Ансамбль', callback_data=f'run:voting:{text}'),
                   InlineKeyboardButton('BiLSTM', callback_data=f'run:bilstm:{text}')]
        kb = InlineKeyboardMarkup([[b] for b in buttons])
        await update.message.reply_text('Выберите модель для этого сообщения:', reply_markup=kb)

    tmp_in.unlink(missing_ok=True)
    tmp_wav.unlink(missing_ok=True)

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    user = q.from_user
    if data.startswith('model:'):
        model = data.split(':', 1)[1]
        set_user(user.id, model=model)
        await q.answer()
        await q.edit_message_text(f'Выбранная модель сохранена: {model}')
        return
    if data.startswith('run:'):
        parts = data.split(':', 2)
        if len(parts) < 3:
            await q.answer('Неправильные данные')
            return
        model = parts[1]
        text = parts[2]
        res = predict_toxicity(text, model)
        await q.answer()
        await q.edit_message_text(f'Результат: {label_to_text(res["label"])} — {res["prob"]*100:.1f}% (модель: {res["model"]})')
        return

def _check_token(token):
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode()
            if '"ok":true' in body:
                print("Token is valid.")
                return True
            print("Token check failed:", body)
            return False
    except urllib.error.URLError as e:
        print("Error connecting to Telegram API:", e)
        return False

def main():
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        print('TELEGRAM_TOKEN not set in environment (put in .env)')
        return

    _check_token(token)

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('model', choose_model_cmd))
    app.add_handler(CommandHandler('mode', mode_cmd))
    app.add_handler(CommandHandler('transcript', transcript_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(CallbackQueryHandler(callback_query_handler))

    print('Starting bot (async polling)...')
    app.run_polling()

if __name__ == '__main__':
    main()