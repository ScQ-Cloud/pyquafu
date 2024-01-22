# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sqlite3
from pathlib import Path


def print_task_info(task):
    """
    Helper function to print task information.
    """
    task_id, group_name, task_name, status, priority, send_time, finish_time = task
    print(f"Task ID: {task_id}")
    print(f"Group Name: {group_name}")
    print(f"Task Name: {task_name}")
    print(f"Status: {status}")
    print(f"Priority: {priority}")
    print(f"Send Time: {send_time}")
    print(f"Finish Time: {finish_time}")
    print("------------------------")


class QuafuTaskDatabase:
    """
    A helper class use sqlite3 database to handle information of quafu tasks at local equipment.
    When initialized and called, it will connect a database file named 'tasks.db' (create new
    one if not exist) at the specified directory. The database will contain a table named 'tasks'
    in which every 'task' has a unique task_id. Other attributes include group name, task name,
    status, priority and send time.

    - Usage:
    Use this class by 'with' statement. For example:
    >>> with QuafuTaskDatabase(db_dir='./') as db:
    ...     db.insert_task(1, "Done", group_name="Group 1", task_name="Task1", priority=2)
    ...     print("Task list:")
    ...     for task_info in db.find_all_tasks():
    ...         print_task_info(task_info)
    This way ensures the database connection is closed and submission committed automatically.
    """

    def __init__(self, db_dir="./"):
        self.database_name = "tasks.db"
        self.database_dir = Path(db_dir)
        self.conn = None

    def __enter__(self):
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
        self.conn = sqlite3.connect(self.database_dir / self.database_name)
        self._create_table()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                group_name TEXT DEFAULT NULL,
                task_name TEXT DEFAULT NULL,
                status TEXT,
                priority INTEGER,
                send_time TIMESTAMP,
                finish_time TIMESTAMP DEFAULT NULL
            )
        """
        )
        cursor.close()

    # region data manipulation
    def insert_task(
        self,
        task_id,
        status,
        send_time: str = None,
        priority=2,
        group_name=None,
        task_name=None,
        finish_time: str = None,
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO tasks "
            "(task_id, group_name, task_name, status, priority, send_time, finish_time) "
            "VALUES "
            "(?, ?, ?, ?, ?, ?, ?)",
            (task_id, group_name, task_name, status, priority, send_time, finish_time),
        )
        cursor.close()

    def delete_task(self, task_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
        cursor.close()
        print(f"Task {task_id} has been deleted from local database")

    def update_task_status(self, task_id, status):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE tasks SET status=? WHERE task_id=?", (status, task_id))
        cursor.close()

    # endregion

    # region fetch tasks
    def find_all_tasks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    def find_by_status(self, status):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE status=?", (status,))
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    def find_by_priority(self, priority):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE priority=?", (priority,))
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    def find_by_group(self, group_name):
        cursor = self.conn.cursor()
        if group_name is None:
            cursor.execute("SELECT * FROM tasks WHERE group_name IS NULL")
        else:
            cursor.execute("SELECT * FROM tasks WHERE group_name=?", (group_name,))
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    def find_by_name(self, task_name):
        cursor = self.conn.cursor()
        if task_name is None:
            cursor.execute("SELECT * FROM tasks WHERE task_name IS NULL")
        else:
            cursor.execute("SELECT * FROM tasks WHERE task_name=?", (task_name,))
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    def find_by_time(self, start_time, end_time):
        """
        get tasks sent between start_time and end_time.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM tasks WHERE send_time BETWEEN ? AND ?",
            (start_time, end_time),
        )
        tasks = cursor.fetchall()
        cursor.close()
        return tasks

    # endregion
