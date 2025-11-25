import requests

with open("tesla_jd.txt", "r", encoding="utf-8") as f:
    jd = f.read()

resp = requests.post(
    "http://127.0.0.1:8000/jd-gap-analysis",
    json={"jd_text": jd}
)

print(resp.status_code)
print(resp.json())
