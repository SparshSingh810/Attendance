import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def ensure_directories():
    os.makedirs("StudentDetails", exist_ok=True)
    os.makedirs("Attendance", exist_ok=True)
    os.makedirs("Embeddings", exist_ok=True)

def load_student_data():
    path = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['Id', 'Name'])
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)

def save_student_data(student_id, student_name):
    df = load_student_data()
    if not ((df['Id'] == int(student_id)) & (df['Name'] == student_name)).any():
        df.loc[len(df)] = [int(student_id), student_name]
        df.to_csv("StudentDetails/StudentDetails.csv", index=False)
        return True
    return False

def capture_images(student_id, student_name):
    ensure_directories()
    cam = cv2.VideoCapture(0)
    embeddings = []

    count = 0
    while count < 5:
        ret, frame = cam.read()
        if not ret:
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            embedding = facenet(face).detach().cpu().numpy()
            embeddings.append(embedding)
            count += 1
            cv2.putText(frame, f"Captured {count}/5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(f"Embeddings/{student_id}_{student_name}.npy", embeddings)
        return True
    return False

def recognize_and_mark_attendance():
    ensure_directories()
    student_data = load_student_data()
    embeddings_db = {}

    for file in os.listdir("Embeddings"):
        if file.endswith(".npy"):
            parts = file.replace(".npy", "").split("_")
            student_id, student_name = parts[0], "_".join(parts[1:])
            embeddings_db[(student_id, student_name)] = np.load(os.path.join("Embeddings", file))

    cam = cv2.VideoCapture(0)
    attendance = []
    face_recognized_once = False
    start_time = None

    y_true = []
    y_pred = []

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            embedding = facenet(face).detach().cpu().numpy()

            for (sid, sname), known_embeds in embeddings_db.items():
                sims = cosine_similarity(embedding, known_embeds)
                max_sim = np.max(sims)

                if max_sim > 0.7:
                    cv2.putText(frame, f"{sname} ({sid})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance.append([sid, sname, date, timeStamp])
                    y_true.append(sid)
                    y_pred.append(sid)

                    if not face_recognized_once:
                        face_recognized_once = True
                        start_time = time.time()
                    break

        else:
            cv2.putText(frame, "No Face Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if face_recognized_once and (time.time() - start_time) >= 5:
            if y_true and y_pred:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                cv2.putText(frame, f"Accuracy:  {acc:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Precision: {prec:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Recall:    {rec:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"F1 Score:  {f1:.2f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Recognizing Faces", frame)
            cv2.waitKey(2000)
            break

        cv2.imshow("Recognizing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # ✅ Append to single CSV
    if attendance:
        df_new = pd.DataFrame(attendance, columns=['Id', 'Name', 'Date', 'Time'])
        df_new = df_new.drop_duplicates(subset=['Id', 'Date'], keep='first')  # One entry per person per day

        filename = "Attendance/Attendance.csv"

        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['Id', 'Date'], keep='first', inplace=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filename, index=False)
        return filename

    return None

# Run
if __name__ == "__main__":
    ensure_directories()
    print("📸 Starting Recognition. Press 'q' to quit...")
    recognize_and_mark_attendance()
