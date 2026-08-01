Here's a compact cheat sheet you can save.

## 🚀 Build & Run Streamlit Dashboard (Docker)

### 1. Rebuild after code changes

```bash
docker compose down
docker compose build
```

(or)

```bash
docker compose up --build -d
```

---

### 2. Open shell inside container

```bash
docker compose exec app bash
```

---

### 3. Verify Streamlit

```bash
streamlit --version
```

---

### 4. Run dashboard

```bash
streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501
```

---

### 5. Open in browser

```
http://localhost:8501
```

---

## 🛑 Stop everything

```bash
docker compose down
```

---

## 🔄 If Docker uses old code

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## 📋 Check container status

```bash
docker ps
```

---

## 📜 View logs

```bash
docker compose logs -f
```

---

## ⚠️ Remember

Your `docker-compose.yml` should expose:

```yaml
ports:
  - "8000:8000"
  - "8501:8501"
```

And keep the Dockerfile's original `CMD` (FastAPI). Run Streamlit manually inside the container when you want to test the dashboard. This lets you use the same image for both the API and the dashboard.
