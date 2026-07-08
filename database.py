# database.py
# 森林昆虫监测系统数据库模块

import sqlite3
from datetime import datetime
import os

DB_NAME = "insect_monitor.db"


# ==========================
# 初始化数据库
# ==========================
def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        detect_time TEXT NOT NULL,

        insect_name TEXT NOT NULL,

        confidence REAL NOT NULL,

        source TEXT,

        image_path TEXT
    )
    """)

    conn.commit()
    conn.close()

    print("✅ 数据库初始化完成")


# ==========================
# 保存识别记录
# ==========================
def save_record(
    insect_name,
    confidence,
    source="camera",
    image_path=""
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO records
    (
        detect_time,
        insect_name,
        confidence,
        source,
        image_path
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        detect_time,
        insect_name,
        confidence,
        source,
        image_path
    ))

    conn.commit()
    conn.close()

    print(
        f"📝 已保存记录: "
        f"{insect_name} "
        f"({confidence:.1f}%)"
    )


# ==========================
# 查看全部记录
# ==========================
def show_all_records():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM records
    ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    if not records:
        print("暂无数据")
        return

    print("\n========== 所有记录 ==========")

    for row in records:
        print(row)


# ==========================
# 获取总记录数
# ==========================
def get_total_count():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM records
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


# ==========================
# 获取昆虫统计
# ==========================
def get_insect_statistics():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT insect_name,
           COUNT(*)
    FROM records
    GROUP BY insect_name
    ORDER BY COUNT(*) DESC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


# ==========================
# 删除全部数据（测试用）
# ==========================
def clear_database():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM records
    """)

    conn.commit()
    conn.close()

    print("⚠️ 数据已清空")


# ==========================
# 测试运行
# ==========================
if __name__ == "__main__":

    print("=" * 50)
    print("森林昆虫监测系统数据库")
    print("=" * 50)

    init_database()

    print(f"\n数据库文件: {os.path.abspath(DB_NAME)}")

    print("\n当前记录数:")
    print(get_total_count())

    print("\n昆虫统计:")
    stats = get_insect_statistics()

    if stats:
        for name, count in stats:
            print(f"{name}: {count}")
    else:
        print("暂无数据")