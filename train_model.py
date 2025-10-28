import os
import gc
import json
import torch
import pickle
import random
import logging
import sacrebleu
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm.auto import tqdm, trange
from torch.utils.data import random_split
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer, AdamW

"""
Скрипт для обучения модели NLLB-200 с поддержкой кабардинского языка.
Поддерживает fine-tuning на параллельных корпусах ru-kbd.
"""

# Опциональный логин в HuggingFace (если требуется доступ к приватным датасетам)
try:
    from huggingface_hub import login
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("Logged in to HuggingFace Hub")
except ImportError:
    print("Warning: huggingface_hub not available")

# ==================== Конфигурация ====================

# ========== Пути ==========
DATA_FOLDER = 'data'
MODEL_RAW_PATH = 'nllb_kbd_raw'  # Путь к модели с расширенным токенизатором
OUTPUT_BASE_PATH = 'models/nllbkbd-3.0-320K'
MODEL_SAVE_PATH = f'{OUTPUT_BASE_PATH}'  # Путь для сохранения обученной модели
LOG_FILE = 'train_model.log'
ERROR_LOG_FILE = 'error_data.txt'
PICKLE_CACHE = 'data.pkl'  # Кеш для разделенных датасетов

# ========== Конфигурация датасетов ==========
# Выберите источник данных: "huggingface" или "json"
DATASET_SOURCE = "json"  # "huggingface" или "json"

# Для HuggingFace датасетов (требует HF_TOKEN для приватных датасетов)
HF_DATASETS = [
    "alimboff/rukbd-63k",
    "alimboff/100K-yandex-kbd",
    "alimboff/12K-kbd"
]

# Для локального JSON файла
JSON_DATASET_PATH = "data/circassian_parallel_ru_kbd.json"  # Путь к JSON файлу с параллельными текстами

# ========== Языковые пары ==========
LANGS = [('ru', 'rus_Cyrl'), ('kbd', 'kbd_Cyrl')]

# ========== Параметры обучения ==========
EPOCHS = 10
BATCH_SIZE = 25
LEARNING_RATE = 0.0005
MAX_LENGTH = 128
VALIDATION_STEP = 1000  # Валидация каждые N шагов
NUM_BEAMS = 4

# ========== Параметры разделения данных ==========
TRAIN_SIZE = 0.9
VAL_SIZE = 0.05
# TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE (автоматически)

# ========== Early Stopping ==========
EARLY_STOPPING_PATIENCE = 3  # Количество эпох без улучшения
MIN_DELTA = 0.0001  # Минимальное улучшение метрики

# ========== Логирование ==========
logger = logging.getLogger('trainModelLogger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info('Logger initialized')

# Создание необходимых директорий
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ==================== Загрузка модели и токенизатора ====================

logger.info(f"Загрузка модели из {MODEL_RAW_PATH}...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_RAW_PATH,
    ignore_mismatched_sizes=True
)
tokenizer = NllbTokenizer.from_pretrained(MODEL_RAW_PATH)

# Проверка и перенос модели на GPU
if torch.cuda.is_available():
    logger.info("CUDA доступна, загружаем модель на GPU")
    model.cuda()
else:
    logger.info("CUDA недоступна, используем CPU")

logger.info("Модель и токенизатор загружены")

# Инициализация метрик
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)

# ==================== Функции загрузки данных ====================

def load_dataset_from_json(json_path: str):
    """
    Загружает параллельный корпус из локального JSON файла.

    Args:
        json_path: Путь к JSON файлу с параллельными текстами

    Returns:
        List[dict]: Список словарей с ключами 'ru' и 'kbd'

    Ожидаемый формат JSON:
    [
        {"ru": "текст на русском", "kbd": "текст на кабардинском"},
        ...
    ]
    """
    logger.info(f"Загрузка датасета из JSON: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON файл не найден: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON должен содержать список объектов, получено: {type(data)}")

    # Валидация данных
    valid_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Элемент {i} не является объектом, пропускаем")
            continue

        if 'ru' not in item or 'kbd' not in item:
            logger.warning(f"Элемент {i} не содержит ключи 'ru' или 'kbd', пропускаем")
            continue

        if not item['ru'] or not item['kbd']:
            logger.warning(f"Элемент {i} содержит пустые значения, пропускаем")
            continue

        valid_data.append({
            'ru': str(item['ru']).strip(),
            'kbd': str(item['kbd']).strip()
        })

    logger.info(f"Загружено {len(valid_data)} валидных записей из {len(data)}")
    return valid_data


def load_datasets_from_huggingface(dataset_names):
    """
    Загружает и объединяет несколько датасетов из HuggingFace.

    Args:
        dataset_names: Список названий датасетов на HuggingFace

    Returns:
        Dataset: Объединенный датасет
    """
    logger.info("Загрузка датасетов из HuggingFace...")
    datasets_list = []

    for dataset_name in dataset_names:
        logger.info(f"  Загрузка {dataset_name}...")
        try:
            ds = load_dataset(dataset_name)
            datasets_list.append(ds['train'])
            logger.info(f"    Загружено {len(ds['train'])} записей")
        except Exception as e:
            logger.error(f"    Ошибка загрузки {dataset_name}: {e}")
            raise

    # Объединяем все датасеты
    dataset = concatenate_datasets(datasets_list)
    logger.info(f"Всего записей: {len(dataset)}")

    return dataset


# ==================== Загрузка и подготовка данных ====================

if os.path.isfile(PICKLE_CACHE):
    logger.info(f"Загрузка кешированных датасетов из {PICKLE_CACHE}...")
    with open(PICKLE_CACHE, 'rb') as file:
        data = pickle.load(file)

    all_train_data = data["all"]["train"]
    all_val_data = data["all"]["val"]
    all_test_data = data["all"]["test"]
    logger.info(f"Датасеты загружены: train={len(all_train_data)}, val={len(all_val_data)}, test={len(all_test_data)}")

else:
    logger.info(f"\n{'='*60}")
    logger.info(f"Загрузка данных из источника: {DATASET_SOURCE}")
    logger.info(f"{'='*60}\n")

    # Загрузка данных в зависимости от источника
    if DATASET_SOURCE == "huggingface":
        if not HF_DATASETS:
            raise ValueError("HF_DATASETS пуст. Укажите список датасетов HuggingFace.")
        dataset = load_datasets_from_huggingface(HF_DATASETS)

    elif DATASET_SOURCE == "json":
        if not JSON_DATASET_PATH:
            raise ValueError("JSON_DATASET_PATH не указан.")

        # Загружаем JSON и конвертируем в формат Dataset
        json_data = load_dataset_from_json(JSON_DATASET_PATH)

        # Создаем Dataset из списка словарей
        dataset = Dataset.from_list(json_data)
        logger.info(f"Датасет создан: {len(dataset)} записей")

    else:
        raise ValueError(f"Неизвестный источник данных: {DATASET_SOURCE}. Используйте 'huggingface' или 'json'")

    # Определяем размеры для train, val и test
    train_size = int(TRAIN_SIZE * len(dataset))
    val_size = int(VAL_SIZE * len(dataset))
    test_size = len(dataset) - train_size - val_size

    logger.info(f"\nРазделение данных:")
    logger.info(f"  Train: {train_size} ({TRAIN_SIZE*100:.0f}%)")
    logger.info(f"  Val:   {val_size} ({VAL_SIZE*100:.0f}%)")
    logger.info(f"  Test:  {test_size} ({(1-TRAIN_SIZE-VAL_SIZE)*100:.0f}%)")

    # Разделяем данные на train, val и test
    all_train_data, all_val_data, all_test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Кешируем разделенные данные
    data = {
        "all": {
            "train": all_train_data,
            "val": all_val_data,
            "test": all_test_data
        }
    }

    logger.info(f"\nСохранение кеша в {PICKLE_CACHE}...")
    with open(PICKLE_CACHE, 'wb') as file:
        pickle.dump(data, file)
    logger.info("Кеш сохранен")

logger.info(f"\nТокен языка kbd_Cyrl ID: {tokenizer.convert_tokens_to_ids(['kbd_Cyrl'])}")

# ==================== Вспомогательные функции ====================

def cleanup():
    """Очищает память GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def get_batch_pairs(batch_size, dataset, step=None):
    """
    Извлекает батч параллельных предложений из датасета.

    Args:
        batch_size: Размер батча
        dataset: Датасет с параллельными текстами
        step: Номер шага (если None, выбираются случайные элементы)

    Returns:
        Tuple: (source_texts, target_texts, source_lang, target_lang)
    """
    # Случайно выбираем направление перевода
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)

    if step is None:
        # Случайная выборка
        indices = random.sample(range(len(dataset)), batch_size)
        batch_data = [dataset[idx] for idx in indices]
    else:
        # Последовательная выборка по шагу
        start_idx = step * batch_size
        end_idx = start_idx + batch_size
        batch_data = [dataset[idx] for idx in range(start_idx, min(end_idx, len(dataset)))]

    # Извлекаем тексты на исходном и целевом языках
    xx = [item[l1] for item in batch_data]
    yy = [item[l2] for item in batch_data]

    return (xx, yy, long1, long2)

def translate(
    text,
    src_lang='kbd_Cyrl',
    tgt_lang='rus_Cyrl',
    a=32,
    b=3,
    num_return_sequences=1,
    max_input_length=1024,
    num_beams=NUM_BEAMS,
    **kwargs
):
    """
    Переводит текст или список текстов с исходного языка на целевой.

    Args:
        text: Строка или список строк для перевода
        src_lang: Код исходного языка (например, 'kbd_Cyrl')
        tgt_lang: Код целевого языка (например, 'rus_Cyrl')
        a: Базовая длина генерируемого текста
        b: Множитель длины генерируемого текста
        num_return_sequences: Количество вариантов перевода
        max_input_length: Максимальная длина входного текста
        num_beams: Количество лучей для beam search
        **kwargs: Дополнительные параметры для model.generate()

    Returns:
        List[str]: Список переводов
    """
    model.eval()
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_length
    )

    result = model.generate(
        **inputs.to('cuda'),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        **kwargs
    )

    return tokenizer.batch_decode(result, skip_special_tokens=True)


# ==================== Функции обучения и тестирования ====================

def train_model(epochs, batch_size, data_train, data_val, save_path):
    """
    Обучает модель на тренировочных данных с валидацией.

    Args:
        epochs: Количество эпох обучения
        batch_size: Размер батча
        data_train: Тренировочный датасет
        data_val: Валидационный датасет
        save_path: Путь для сохранения лучшей модели

    Returns:
        None
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Начало обучения: epochs={epochs}, batch_size={batch_size}")
    logger.info(f"Сохранение модели в: {save_path}")
    logger.info(f"{'='*60}\n")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = (len(data_train) // batch_size) * epochs
    num_warmup_steps = len(data_train) // batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    cleanup()
    x, y, loss, loss_val = None, None, None, None

    losses = []
    bleu_scores = []
    chrf_scores = []

    best_bleu = None
    epochs_no_improve = 0
    
    def run_validation():
        model.eval()
        val_losses = []
        all_references = []
        all_hypotheses = []
        
        tq_val = trange(len(data_val) // batch_size, leave=False)
        with torch.no_grad():
            mean_train_val_loss = None
            
            for i in tq_val:
                try:
                    xx, yy, lang1, lang2 = get_batch_pairs(batch_size, dataset=data_val, step=i)

                    tokenizer.src_lang = lang1
                    x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to('cuda')
                    tokenizer.src_lang = lang2
                    y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to('cuda')

                    outputs = model.generate(**x)
                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    y_masked = y.input_ids.clone()
                    y_masked[y_masked == tokenizer.pad_token_id] = -100
                    
                    loss_val = model(**x, labels=y_masked).loss
                    val_losses.append(loss_val.item())

                    decoded_labels = tokenizer.batch_decode(y.input_ids, skip_special_tokens=True)

                    all_hypotheses.extend(decoded_preds)
                    all_references.extend(decoded_labels)

                    mean_train_val_loss = sum(val_losses) / len(val_losses)
                    tq_val.set_description(f'Validation | val_loss={mean_train_val_loss:.4f}')

                except ValueError as e:
                    logger.error(f'DATA ERROR | {xx}, {yy}, {lang1}, {lang2}')
                    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as file:
                        file.write(f'{xx}, {yy}, {lang1}, {lang2}\n')
        
        bleu_score = bleu_calc.corpus_score(all_hypotheses, [all_references]).score
        chrf_score = chrf_calc.corpus_score(all_hypotheses, [all_references]).score
        
        return mean_train_val_loss, bleu_score, chrf_score

    # Основной цикл обучения
    for epoch in range(epochs):
        model.train()
        tq_train = trange(len(data_train) // batch_size, leave=False)
        mean_train_loss = None

        for i in tq_train:
            xx, yy, lang1, lang2 = get_batch_pairs(batch_size, dataset=data_train)

            try:
                tokenizer.src_lang = lang1
                x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to('cuda')
                tokenizer.src_lang = lang2
                y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to('cuda')
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                loss = model(**x, labels=y.input_ids).loss

                optimizer.zero_grad(set_to_none=True)
                loss = loss.mean()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                scheduler.step()

            except ValueError as e:
                logger.error(f'DATA ERROR | {e}')
                with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as file:
                    file.write(f'{xx}, {yy}, {lang1}, {lang2}\n')

            except RuntimeError as e:
                optimizer.zero_grad(set_to_none=True)
                x, y, loss = None, None, None
                cleanup()
                # logger.error(f'Runtime error: max_len={max(len(s) for s in xx + yy)}, {e}')
                continue

            mean_train_loss = sum(losses) / len(losses)

            tq_train.set_description(
                f'Epoch [{epoch+1}/{epochs}] | train_loss={mean_train_loss:.4f} | LR {optimizer.param_groups[0]["lr"]:.10f}'
            )

            # Валидация каждые VALIDATION_STEP шагов
            if (i + 1) % VALIDATION_STEP == 0:
                mean_val_loss, bleu_score, chrf_score = run_validation()
                bleu_scores.append(bleu_score)
                chrf_scores.append(chrf_score)

                if best_bleu is None or bleu_score > best_bleu + MIN_DELTA:
                    best_bleu = bleu_score
                    epochs_no_improve = 0
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(
                        f'✓ Saved model [{epoch+1}/{epochs}] Step [{i+1}] | '
                        f'train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | '
                        f'BLEU={bleu_score:.4f} | chrF={chrf_score:.4f} | '
                        f'LR {optimizer.param_groups[0]["lr"]:.10f}'
                    )
                else:
                    logger.info(
                        f'Epoch [{epoch+1}/{epochs}] Step [{i+1}] | '
                        f'train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | '
                        f'BLEU={bleu_score:.4f} | chrF={chrf_score:.4f} | '
                        f'LR {optimizer.param_groups[0]["lr"]:.10f}'
                    )

                model.train()

        # Валидация в конце эпохи
        mean_val_loss, bleu_score, chrf_score = run_validation()
        bleu_scores.append(bleu_score)
        chrf_scores.append(chrf_score)

        if best_bleu is None or bleu_score > best_bleu + MIN_DELTA:
            epochs_no_improve = 0
            best_bleu = bleu_score
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(
                f'✓ Saved model [{epoch+1}/{epochs}] End epoch | '
                f'train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | '
                f'BLEU={bleu_score:.4f} | chrF={chrf_score:.4f} | '
                f'LR {optimizer.param_groups[0]["lr"]:.10f}'
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f'Epoch [{epoch+1}/{epochs}] End epoch | '
                f'train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | '
                f'BLEU={bleu_score:.4f} | chrF={chrf_score:.4f} | '
                f'LR {optimizer.param_groups[0]["lr"]:.10f}'
            )

        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f'Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs')
            break

    logger.info(f"Обучение завершено! Модель сохранена в {save_path}")

def test_model(batch_size, data_test, model_path):
    """
    Тестирует обученную модель на тестовых данных.

    Args:
        batch_size: Размер батча
        data_test: Тестовый датасет
        model_path: Путь к обученной модели

    Returns:
        None
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Тестирование модели из {model_path}")
    logger.info(f"{'='*60}\n")

    test_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    test_tokenizer = NllbTokenizer.from_pretrained(model_path)

    if torch.cuda.is_available():
        test_model.cuda()

    test_losses = []
    all_references = []
    all_hypotheses = []

    tq_test = trange(len(data_test) // batch_size, leave=False)

    test_model.eval()
    with torch.no_grad():
        for i in tq_test:
            xx, yy, lang1, lang2 = get_batch_pairs(batch_size, dataset=data_test, step=i)

            try:
                # Токенизация входных данных
                test_tokenizer.src_lang = lang1
                x = test_tokenizer(
                    xx, return_tensors='pt', padding=True,
                    truncation=True, max_length=MAX_LENGTH
                ).to('cuda')

                test_tokenizer.src_lang = lang2
                y = test_tokenizer(
                    yy, return_tensors='pt', padding=True,
                    truncation=True, max_length=MAX_LENGTH
                ).to('cuda')

                # Генерация предсказаний
                outputs = test_model.generate(**x)
                decoded_preds = test_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Маскировка паддинговых токенов для вычисления потерь
                y_masked = y.input_ids.clone()
                y_masked[y_masked == test_tokenizer.pad_token_id] = -100

                # Расчет потерь
                loss_test = test_model(**x, labels=y_masked).loss
                test_losses.append(loss_test.item())

                # Декодирование целевых меток
                decoded_labels = test_tokenizer.batch_decode(y.input_ids, skip_special_tokens=True)

                # Сбор предсказаний и целевых меток
                all_hypotheses.extend(decoded_preds)
                all_references.extend(decoded_labels)

                mean_test_loss = sum(test_losses) / len(test_losses)
                tq_test.set_description(f'TEST | test_loss={mean_test_loss:.4f}')

            except ValueError as e:
                logger.error(f'DATA ERROR | {e}')
                with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as file:
                    file.write(f'{xx}, {yy}, {lang1}, {lang2}\n')

    # Вычисление финальных метрик
    bleu_score = bleu_calc.corpus_score(all_hypotheses, [all_references]).score
    chrf_score = chrf_calc.corpus_score(all_hypotheses, [all_references]).score

    logger.info(
        f'TEST RESULTS | test_loss={mean_test_loss:.4f} | '
        f'BLEU={bleu_score:.4f} | chrF={chrf_score:.4f} | '
        f'Model: {model_path}'
    )


# ==================== Запуск обучения и тестирования ====================

if __name__ == "__main__":
    # Обучение модели
    train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        data_train=all_train_data,
        data_val=all_val_data,
        save_path=MODEL_SAVE_PATH
    )

    # Тестирование обученной модели
    test_model(
        batch_size=BATCH_SIZE,
        data_test=all_test_data,
        model_path=MODEL_SAVE_PATH
    )