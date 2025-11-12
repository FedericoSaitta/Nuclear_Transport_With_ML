import sqlite3
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

class SQLiteLogger(Logger):
    def __init__(self, db_path='metrics.db', name='experiment'):
        super().__init__()
        self.db_path = db_path
        self._name = name
        self._version = 0
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database and create table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                experiment_name TEXT,
                version INTEGER,
                step INTEGER,
                epoch INTEGER,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    @property
    def name(self):
        return self._name
    
    @property
    def version(self):
        return self._version
    
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        """Log metrics to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
                INSERT INTO metrics (experiment_name, version, step, epoch, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.name, self.version, step, metrics.get('epoch', None), metric_name, float(metric_value)))
        
        conn.commit()
        conn.close()
    
    @rank_zero_only
    def log_hyperparams(self, params):
        """Optionally log hyperparameters"""
        pass
    
    def save(self):
        """Optional: implement save logic"""
        pass
    
    def finalize(self, status):
        """Optional: implement finalization logic"""
        pass