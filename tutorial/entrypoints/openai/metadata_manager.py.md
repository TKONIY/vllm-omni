# `metadata_manager.py` — 元数据管理器

## 文件概述

`MetadataManager` 为语音样本（说话人）和缓存信息提供统一的持久化管理。它通过文件锁实现跨进程的并发安全元数据读写，支持原子性的读-修改-写操作。

## 关键代码解析

### 跨进程安全的原子更新

```python
def _update_with_file_lock(self, update_fn):
    """使用文件锁实现原子性的读-修改-写"""
    lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # 获取排他锁
        metadata = self._load_from_disk()      # 读取最新数据
        result = update_fn(metadata)           # 应用更新
        self._save_to_disk(metadata)           # 持久化
        self._metadata = metadata              # 更新内存缓存
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
```

双重锁机制：
- `threading.Lock`: 线程内并发安全
- `fcntl.flock`: 跨进程并发安全

### 原子写入

```python
def _save_to_disk(self, metadata):
    tmp = self.metadata_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp.replace(self.metadata_file)  # 原子替换
```

使用临时文件 + `replace` 模式确保写入的原子性，避免写入中途崩溃导致数据损坏。

### 说话人管理

```python
def create_speaker(self, speaker_key, speaker_data) -> bool:
    """创建新的说话人条目（去重检查）"""

def update_speaker(self, speaker_key, updates) -> bool:
    """合并更新说话人信息"""

def delete_speaker(self, speaker_key) -> dict | None:
    """删除说话人并清理关联的音频和缓存文件"""

def update_cache_info(self, speaker_key, cache_file_path, status="ready"):
    """更新说话人的缓存状态"""
```

## 核心类/函数

| 名称 | 类型 | 用途 |
|------|------|------|
| `MetadataManager` | 类 | 说话人元数据的并发安全管理 |
| `_update_with_file_lock()` | 方法 | 文件锁保护的原子更新 |
| `create_speaker()` | 方法 | 创建说话人 |
| `update_speaker()` | 方法 | 更新说话人信息（合并语义） |
| `delete_speaker()` | 方法 | 删除说话人及关联文件 |
| `update_cache_info()` | 方法 | 更新缓存状态 |
| `get_uploaded_speakers()` | 方法 | 获取所有说话人 |
| `reload_from_disk()` | 方法 | 强制从磁盘重载 |

## 与其他模块的关系

- 被 `serving_speech.py` 用于管理上传的说话人样本
- 元数据文件存储在 `storage.py` 管理的存储目录中

## 总结

`MetadataManager` 是一个为 TTS 说话人管理设计的并发安全持久化层，通过双重锁机制和原子文件写入保证在多进程环境下的数据一致性。
