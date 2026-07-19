import sqlite3
import json
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "voiceguard.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create the cases table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            customer_name TEXT,
            phone_number TEXT,
            branch TEXT,
            account_ref TEXT,
            fraud_score REAL,
            confidence REAL,
            verdict TEXT,
            bio_features TEXT,
            deep_features TEXT,
            checklist_state TEXT,
            notes TEXT,
            frozen INTEGER DEFAULT 0,
            escalated INTEGER DEFAULT 0,
            otp_code TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_case(case_id, customer_name, phone_number, branch, account_ref, 
                fraud_score, confidence, verdict, bio_features, deep_features, otp_code=None):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Initial checklist state
    default_checklist = json.dumps({
        "chk-id": False,
        "chk-delay": False,
        "chk-otp": False,
        "chk-video": False,
        "chk-escalate": False
    })
    
    cursor.execute("""
        INSERT OR REPLACE INTO cases (
            case_id, timestamp, customer_name, phone_number, branch, account_ref,
            fraud_score, confidence, verdict, bio_features, deep_features, checklist_state, notes, otp_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
    """, (
        case_id, timestamp, customer_name, phone_number, branch, account_ref,
        fraud_score, confidence, verdict, json.dumps(bio_features), json.dumps(deep_features), default_checklist, otp_code
    ))
    conn.commit()
    conn.close()

def update_checklist(case_id, checklist_state):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE cases SET checklist_state = ? WHERE case_id = ?", (json.dumps(checklist_state), case_id))
    conn.commit()
    conn.close()

def update_case_action(case_id, frozen=None, escalated=None, notes=None):
    conn = get_connection()
    cursor = conn.cursor()
    if frozen is not None:
        cursor.execute("UPDATE cases SET frozen = ? WHERE case_id = ?", (1 if frozen else 0, case_id))
    if escalated is not None:
        cursor.execute("UPDATE cases SET escalated = ? WHERE case_id = ?", (1 if escalated else 0, case_id))
    if notes is not None:
        cursor.execute("UPDATE cases SET notes = ? WHERE case_id = ?", (notes, case_id))
    conn.commit()
    conn.close()

def get_case(case_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_cases(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def compute_cosine_similarity(v1, v2):
    keys = ["Jitter_Mean", "Shimmer_Mean", "HNR_Mean", "Pitch_Std", "Flatness_Mean"]
    # Normalizing weights for distance metrics
    scales = {
        "Jitter_Mean": 1000.0,
        "Shimmer_Mean": 100.0,
        "HNR_Mean": 1.0,
        "Pitch_Std": 1.0,
        "Flatness_Mean": 10000.0
    }
    
    vec1 = [v1.get(k, 0.0) * scales[k] for k in keys]
    vec2 = [v2.get(k, 0.0) * scales[k] for k in keys]
    
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def get_similar_cases(target_case_id, limit=3):
    target = get_case(target_case_id)
    if not target:
        return []
    try:
        target_features = json.loads(target["bio_features"])
    except:
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE case_id != ? AND case_id NOT LIKE 'VG-SEED%'", (target_case_id,))
    # Fallback to seeded cases if there are no other active runs
    rows = cursor.fetchall()
    if len(rows) < 5:
        cursor.execute("SELECT * FROM cases WHERE case_id != ?", (target_case_id,))
        rows = cursor.fetchall()
    conn.close()
    
    scored_cases = []
    for row in rows:
        case_data = dict(row)
        try:
            feats = json.loads(case_data["bio_features"])
            similarity = compute_cosine_similarity(target_features, feats)
            case_data["similarity_score"] = round(similarity * 100, 1)
            scored_cases.append(case_data)
        except Exception as e:
            continue
            
    scored_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_cases[:limit]

def seed_database():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if table already contains data
    cursor.execute("SELECT COUNT(*) FROM cases")
    count = cursor.fetchone()[0]
    if count > 5:
        conn.close()
        return
        
    names = ["Amit Sharma", "Rajesh Patel", "Priya Singh", "Rohan Verma", "Sneha Rao", "Vikram Gupta", 
             "Deepak Reddy", "Ananya Nair", "Suresh Kumar", "Meera Joshi", "Aditya Sen", "Sunita Devi"]
    branches = ["Kolkata Head Office", "Delhi Connaught Place", "Mumbai Nariman Point", "Lucknow Hazratganj", 
                "Chennai Anna Salai", "Bengaluru MG Road", "Hyderabad Banjara Hills"]
    verdicts = ["CLEAR", "LOW_RISK", "MODERATE", "HIGH_RISK", "CRITICAL"]
    
    now = datetime.utcnow()
    for i in range(40):
        case_id = f"VG-SEED{1000 + i}"
        customer_name = random.choice(names)
        phone_number = f"+91 {random.randint(7000, 9999)} {random.randint(100000, 999900)}"
        branch = random.choice(branches)
        account_ref = f"UCO-{random.randint(10000000, 99999999)}"
        
        verd = random.choice(verdicts)
        if verd == "CLEAR":
            fraud_score = random.uniform(0.0, 9.9)
        elif verd == "LOW_RISK":
            fraud_score = random.uniform(10.0, 24.9)
        elif verd == "MODERATE":
            fraud_score = random.uniform(25.0, 49.9)
        elif verd == "HIGH_RISK":
            fraud_score = random.uniform(50.0, 74.9)
        else:
            fraud_score = random.uniform(75.0, 100.0)
            
        confidence = random.uniform(70.0, 99.9)
        
        bio_features = {
            "Jitter_Mean": random.uniform(0.001, 0.015) if verd != "CRITICAL" else random.uniform(0.0001, 0.002),
            "Shimmer_Mean": random.uniform(0.01, 0.08) if verd != "CRITICAL" else random.uniform(0.001, 0.015),
            "HNR_Mean": random.uniform(12.0, 25.0) if verd != "CRITICAL" else random.uniform(28.0, 36.0),
            "Pitch_Std": random.uniform(8.0, 45.0) if verd != "CRITICAL" else random.uniform(0.5, 4.8),
            "Flatness_Mean": random.uniform(0.002, 0.02) if verd != "CRITICAL" else random.uniform(0.00005, 0.0009)
        }
        
        deep_features = [random.uniform(-1.0, 1.0) for _ in range(32)]
        
        delta_days = random.randint(0, 30)
        delta_hours = random.randint(0, 23)
        delta_mins = random.randint(0, 59)
        case_time = now - timedelta(days=delta_days, hours=delta_hours, minutes=delta_mins)
        timestamp = case_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        checklist = {
            "chk-id": random.choice([True, False]),
            "chk-delay": random.choice([True, False]),
            "chk-otp": random.choice([True, False]),
            "chk-video": random.choice([True, False]),
            "chk-escalate": random.choice([True, False])
        }
        
        frozen = 1 if verd in ["HIGH_RISK", "CRITICAL"] and random.choice([True, False]) else 0
        escalated = 1 if verd in ["MODERATE", "HIGH_RISK"] and random.choice([True, False]) else 0
        notes = "Suspicious activity detected. Auto-held." if frozen else "Verified and processed."
        
        cursor.execute("""
            INSERT OR REPLACE INTO cases (
                case_id, timestamp, customer_name, phone_number, branch, account_ref,
                fraud_score, confidence, verdict, bio_features, deep_features, checklist_state, notes, frozen, escalated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id, timestamp, customer_name, phone_number, branch, account_ref,
            round(fraud_score, 1), round(confidence, 1), verd, json.dumps(bio_features), json.dumps(deep_features), 
            json.dumps(checklist), notes, frozen, escalated
        ))
        
    conn.commit()
    conn.close()
    print("Database successfully seeded.")

if __name__ == "__main__":
    seed_database()
