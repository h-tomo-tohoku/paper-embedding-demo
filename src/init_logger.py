import logging
import os
import sys


def init_logger(
    logger_name: str,
    log_file_path: str = "log/info.log",
) -> logging.Logger:
    """loggerの初期化
        標準出力とファイルにDEBUG以上のログを出力する

    Parameters
    ----------
    logger_name : str
        logger
    log_file_path : str, optional
        logの出力パス, by default "info.log"

    Returns
    -------
    logging.Logger
        logger
    """

    os.makedirs("log", exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # DEBUG 以上のログを記録

    # フォーマットの定義
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # ファイルハンドラーの作成（ログをファイルに保存）
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # すべてのログを保存
    file_handler.setFormatter(formatter)

    # コンソールハンドラーの作成（ログをコンソールに出力）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # INFO 以上のログを表示
    console_handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    logger = init_logger("test_logger", "log/test.log")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
