import sentencepiece.sentencepiece_model_pb2 as sp_pb2_model
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
import sentencepiece as spm
import shutil
from typing import List, Optional
import json
import os

# Опциональный импорт datasets (для загрузки из HuggingFace)
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")

"""
Скрипт для расширения токенизатора NLLB-200 с добавлением поддержки кабардинского языка (kbd_Cyrl).
Обучает новую SentencePiece модель на кабардинском корпусе и интегрирует её в существующий токенизатор NLLB.

Поддерживает два способа загрузки корпуса:
1. HuggingFace Dataset (указать dataset_name и column_name)
2. Локальный JSON файл (указать json_path и json_key)
"""

# ==================== Конфигурация ====================

# Название базовой модели NLLB
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ========== Конфигурация корпуса ==========
# Выберите один из двух вариантов загрузки корпуса:

# Вариант 1: HuggingFace Dataset
CORPUS_SOURCE = "json"  # "huggingface" или "json"
DATASET_NAME: Optional[str] = None  # Например: "username/dataset-name"
DATASET_SPLIT: str = "train"  # Например: "train", "test", "validation"
COLUMN_NAME: Optional[str] = None  # Название столбца с текстом на целевом языке

# Вариант 2: Локальный JSON файл
JSON_PATH: str = "data/circassian_parallel_ru_kbd.json"  # Путь к JSON файлу
JSON_KEY: str = "kbd"  # Ключ для извлечения текста (например, "kbd", "en", "text")

# ========== Пути для сохранения ==========
DATA_FOLDER = 'data'
KBD_TEXT_CORPUS = f'{DATA_FOLDER}/kbd_text_plain.txt'  # Сгенерированный корпус

MODEL_OUTPUT_PATH = 'nllb_kbd_raw'  # Финальная модель и токенизатор
SPM_PREFIX = 'tokenizerfiles/spm_temp/SPMKBD'  # Промежуточная SPM модель
NEW_SPM_PATH = 'tokenizerfiles/spm_nllb_kbd.model'  # Итоговая SPM модель

# ========== Параметры языка ==========
NEW_LANG_CODE = 'kbd_Cyrl'

# ========== Параметры SentencePiece ==========
SPM_VOCAB_SIZE = 2**10  # 2^9
SPM_CHARACTER_COVERAGE = 1.0

# ==================== Функции загрузки корпуса ====================

def load_corpus_from_json(json_path: str, key: str) -> List[str]:
    """
    Загружает корпус из локального JSON файла.

    Args:
        json_path: Путь к JSON файлу
        key: Ключ для извлечения текста из каждого объекта

    Returns:
        Список текстов
    """
    print(f"Загрузка корпуса из JSON: {json_path}")
    print(f"Извлечение текста по ключу: '{key}'")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON файл должен содержать список объектов, получено: {type(data)}")

    texts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Элемент {i} не является объектом, пропускаем")
            continue

        if key not in item:
            print(f"Warning: Ключ '{key}' не найден в элементе {i}, пропускаем")
            continue

        text = item[key]
        if text and isinstance(text, str):
            texts.append(text.strip())

    print(f"Загружено {len(texts)} текстов из {len(data)} записей")
    return texts


def load_corpus_from_huggingface(dataset_name: str, split: str, column: str) -> List[str]:
    """
    Загружает корпус из HuggingFace Dataset.

    Args:
        dataset_name: Название датасета на HuggingFace
        split: Сплит датасета (train, test, validation)
        column: Название столбца с текстом

    Returns:
        Список текстов
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "Библиотека 'datasets' не установлена. "
            "Установите её командой: pip install datasets"
        )

    print(f"Загрузка датасета из HuggingFace: {dataset_name}")
    print(f"Сплит: {split}, Столбец: {column}")

    dataset = load_dataset(dataset_name, split=split)

    if column not in dataset.column_names:
        raise ValueError(
            f"Столбец '{column}' не найден в датасете. "
            f"Доступные столбцы: {dataset.column_names}"
        )

    texts = [item[column].strip() for item in dataset if item[column]]

    print(f"Загружено {len(texts)} текстов")
    return texts


def prepare_corpus(
    source: str,
    output_path: str,
    json_path: Optional[str] = None,
    json_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_split: str = "train",
    column_name: Optional[str] = None,
) -> str:
    """
    Подготавливает текстовый корпус для обучения SentencePiece.

    Args:
        source: Источник данных ("json" или "huggingface")
        output_path: Путь для сохранения текстового файла
        json_path: Путь к JSON файлу (для source="json")
        json_key: Ключ для извлечения текста (для source="json")
        dataset_name: Название датасета (для source="huggingface")
        dataset_split: Сплит датасета (для source="huggingface")
        column_name: Название столбца (для source="huggingface")

    Returns:
        Путь к созданному файлу корпуса
    """
    print("\n" + "="*60)
    print("Подготовка корпуса")
    print("="*60)

    # Загрузка текстов
    if source == "json":
        if not json_path or not json_key:
            raise ValueError("Для source='json' необходимо указать json_path и json_key")
        texts = load_corpus_from_json(json_path, json_key)

    elif source == "huggingface":
        if not dataset_name or not column_name:
            raise ValueError("Для source='huggingface' необходимо указать dataset_name и column_name")
        texts = load_corpus_from_huggingface(dataset_name, dataset_split, column_name)

    else:
        raise ValueError(f"Неизвестный источник: {source}. Используйте 'json' или 'huggingface'")

    # Сохранение в текстовый файл
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nСохранение корпуса в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    print(f"Корпус сохранен: {len(texts)} строк")
    print("="*60 + "\n")

    return output_path


# ==================== Подготовка корпуса ====================

# Подготовка текстового корпуса для SentencePiece
prepare_corpus(
    source=CORPUS_SOURCE,
    output_path=KBD_TEXT_CORPUS,
    json_path=JSON_PATH if CORPUS_SOURCE == "json" else None,
    json_key=JSON_KEY if CORPUS_SOURCE == "json" else None,
    dataset_name=DATASET_NAME if CORPUS_SOURCE == "huggingface" else None,
    dataset_split=DATASET_SPLIT if CORPUS_SOURCE == "huggingface" else None,
    column_name=COLUMN_NAME if CORPUS_SOURCE == "huggingface" else None,
)

# ==================== Инициализация ====================

print(f"Загрузка токенизатора из {MODEL_NAME}...")
tokenizer_old = NllbTokenizer.from_pretrained(MODEL_NAME)

# ==================== Обучение SentencePiece ====================

print(f"\nОбучение SentencePiece модели на корпусе {KBD_TEXT_CORPUS}...")
print(f"Размер словаря: {SPM_VOCAB_SIZE} токенов")

# Создаем директорию для SPM файлов
os.makedirs(os.path.dirname(SPM_PREFIX), exist_ok=True)

spm.SentencePieceTrainer.train(
    input=KBD_TEXT_CORPUS,
    model_prefix=SPM_PREFIX,
    vocab_size=SPM_VOCAB_SIZE,
    character_coverage=SPM_CHARACTER_COVERAGE,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=512,
    max_sentence_length=4192 * 4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
)

print("SentencePiece модель обучена успешно!")

# ==================== Объединение токенизаторов ====================

print("\nЗагрузка обученной SentencePiece модели...")
sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())

print("Загрузка оригинальной NLLB SentencePiece модели...")
old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer_old.sp_model.serialized_model_proto())

print("Добавление новых токенов в NLLB модель...")
nllb_tokens_set = {p.piece for p in old_spm.pieces if p.type == 1}

prev_min_score = old_spm.pieces[-1].score
added_tokens_count = 0

for p in added_spm.pieces:
    piece = p.piece
    if p.type != 1:  # Пропускаем не-токены (control symbols и т.д.)
        continue
    if piece not in nllb_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = p.score + prev_min_score  # Более низкий приоритет для новых токенов
        old_spm.pieces.append(new_p)
        added_tokens_count += 1

print(f"Добавлено {added_tokens_count} новых токенов")

print(f"\nСохранение обновленной SentencePiece модели в {NEW_SPM_PATH}...")
os.makedirs(os.path.dirname(NEW_SPM_PATH), exist_ok=True)
with open(NEW_SPM_PATH, 'wb') as f:
    f.write(old_spm.SerializeToString())

# ==================== Создание нового токенизатора ====================

def update_nllb_tokenizer(
    old_tokenizer: NllbTokenizer,
    new_spm_path: str,
    new_lang_codes: List[str],
    output_dir: str
) -> NllbTokenizer:
    """
    Создает новый токенизатор NLLB с обновленной SentencePiece моделью и новыми языковыми кодами.

    Args:
        old_tokenizer: Исходный токенизатор NLLB
        new_spm_path: Путь к новой SentencePiece модели
        new_lang_codes: Список новых языковых кодов для добавления
        output_dir: Директория для сохранения токенизатора

    Returns:
        Обновленный токенизатор NLLB
    """
    print(f"\nСоздание нового токенизатора...")

    # Сохраняем старый токенизатор во временную директорию
    os.makedirs(output_dir, exist_ok=True)
    old_tokenizer.save_pretrained(output_dir)

    # Обновляем конфигурацию токенизатора
    config_path = f"{output_dir}/tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Оставляем только базовые специальные токены (pad, eos, unk, bos)
    cfg["added_tokens_decoder"] = {
        k: v for k, v in cfg["added_tokens_decoder"].items()
        if k in ["0", "1", "2", "3"]
    }
    cfg["additional_special_tokens"] = []

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Удаляем старые файлы токенов и заменяем SentencePiece модель
    os.remove(f"{output_dir}/added_tokens.json")
    os.remove(f"{output_dir}/special_tokens_map.json")
    os.remove(f"{output_dir}/sentencepiece.bpe.model")
    shutil.copy(new_spm_path, f"{output_dir}/sentencepiece.bpe.model")

    # Загружаем новый токенизатор с обновленными языковыми кодами
    new_tokenizer = NllbTokenizer.from_pretrained(
        output_dir,
        additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + new_lang_codes),
    )

    print(f"Токенизатор создан! Размер словаря: {len(new_tokenizer)}")
    return new_tokenizer


print("\nОбновление токенизатора NLLB...")
tokenizer = update_nllb_tokenizer(
    tokenizer_old,
    NEW_SPM_PATH,
    [NEW_LANG_CODE],
    MODEL_OUTPUT_PATH
)

# Статистика по токенизатору
added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
print(f"\nСтатистика токенизатора:")
print(f"  Размер словаря: {len(tokenizer)}")
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Добавлено токенов: {len(added_vocab)}")
print(f"  Код языка {NEW_LANG_CODE} ID: {tokenizer.convert_tokens_to_ids([NEW_LANG_CODE])}")

# ==================== Расширение модели ====================

print(f"\nЗагрузка базовой модели {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"Размер embedding слоя до расширения: {model.model.shared.num_embeddings}")

# Расширяем embedding слой модели под новый размер словаря
model.resize_token_embeddings(len(tokenizer))
print(f"Размер embedding слоя после расширения: {model.model.shared.num_embeddings}")

# Инициализация весов для нового языкового кода (копируем из русского)
print(f"\nИнициализация весов для {NEW_LANG_CODE} из rus_Cyrl...")
model.model.shared.weight.data[
    tokenizer.convert_tokens_to_ids([NEW_LANG_CODE])
] = model.model.shared.weight.data[
    tokenizer_old.convert_tokens_to_ids(['rus_Cyrl'])
]

# Копируем веса для всех языковых кодов и <mask>
print("Копирование весов для языковых кодов...")
moved_tokens = list(tokenizer_old.lang_code_to_id.keys()) + ['<mask>']
model.model.shared.weight.data[
    tokenizer.convert_tokens_to_ids(moved_tokens)
] = model.model.shared.weight.data[
    tokenizer_old.convert_tokens_to_ids(moved_tokens)
]

# Инициализация весов для новых токенов (усреднением весов их подтокенов)
print(f"Инициализация весов для {len(added_vocab)} новых токенов...")
for token in tqdm(added_vocab, desc="Инициализация embeddings"):
    # Токенизируем новый токен старым токенизатором
    subtokens = tokenizer_old(token, add_special_tokens=False).input_ids
    if len(subtokens) == 0:
        subtokens = [tokenizer_old.unk_token_id]

    # Усредняем веса подтокенов
    token_idx = tokenizer.convert_tokens_to_ids(token)
    model.model.shared.weight.data[token_idx] = model.model.shared.weight.data[subtokens].mean(0)

print(f"Инициализация весов завершена!")

# ==================== Сохранение ====================

print(f"\nСохранение модели и токенизатора в {MODEL_OUTPUT_PATH}...")
model.save_pretrained(MODEL_OUTPUT_PATH)
tokenizer.save_pretrained(MODEL_OUTPUT_PATH)

print("\n" + "="*60)
print("Готово! Модель и токенизатор успешно сохранены.")
print("="*60)