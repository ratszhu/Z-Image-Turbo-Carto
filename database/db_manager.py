# -*- coding: utf-8 -*-
"""
数据库管理器 (SQLite)
负责生成记录的增删改查。
"""
import sqlite3
import json
import os
from datetime import datetime
import config

class DatabaseManager:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prompt TEXT,
                negative_prompt TEXT,
                steps INTEGER,
                cfg REAL,
                seed INTEGER,
                width INTEGER,
                height INTEGER,
                lora_enabled INTEGER,
                lora_scale REAL,
                lora_id TEXT,
                lora_name TEXT,
                device TEXT,
                duration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 旧版数据库无损升级：保留全部历史记录。
        existing_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(generations)").fetchall()
        }
        if "lora_id" not in existing_columns:
            cursor.execute("ALTER TABLE generations ADD COLUMN lora_id TEXT")
        if "lora_name" not in existing_columns:
            cursor.execute("ALTER TABLE generations ADD COLUMN lora_name TEXT")
        conn.commit()
        conn.close()

    def add_record(self, record_data):
        """
        添加一条生成记录
        Args:
            record_data (dict): 包含生成参数的字典
        Returns:
            int: 新记录的 ID
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        query = '''
            INSERT INTO generations (
                filename, prompt, negative_prompt, steps, cfg, seed, 
                width, height, lora_enabled, lora_scale, lora_id, lora_name, device, duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        values = (
            record_data.get('filename'),
            record_data.get('prompt'),
            record_data.get('negative_prompt'),
            record_data.get('steps'),
            record_data.get('cfg'),
            record_data.get('seed'),
            record_data.get('width'),
            record_data.get('height'),
            1 if record_data.get('lora_enabled') else 0,
            record_data.get('lora_scale'),
            record_data.get('lora_id'),
            record_data.get('lora_name'),
            record_data.get('device'),
            record_data.get('duration')
        )
        
        cursor.execute(query, values)
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return new_id

    def get_history(self, limit=50, offset=0):
        """获取历史记录列表 (按时间倒序)"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row # 让结果可以通过列名访问
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM generations 
            ORDER BY id DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 转换为字典列表
        results = []
        for row in rows:
            item = dict(row)
            # 转换布尔值
            item['lora_enabled'] = bool(item['lora_enabled'])
            results.append(item)
            
        return results

    def delete_record(self, record_id):
        """删除记录及对应的物理文件"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. 先查询文件名
        cursor.execute('SELECT filename FROM generations WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        
        if row:
            filename = row['filename']
            file_path = os.path.join(config.OUTPUT_DIR, filename)
            
            # 2. 删除物理文件
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ [DB] 文件已删除: {filename}")
                except Exception as e:
                    print(f"⚠️ [DB] 文件删除失败: {e}")
            
            # 3. 删除数据库记录
            cursor.execute('DELETE FROM generations WHERE id = ?', (record_id,))
            conn.commit()
            success = True
        else:
            success = False
            
        conn.close()
        return success
