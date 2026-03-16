import os
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path


def load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ.setdefault(key, value)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv_file(ROOT_DIR / ".env")

SENDER = os.getenv("SMTP_SENDER")
APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
RECIPIENT = os.getenv("SMTP_RECIPIENT")

FILES = [
    "images/Education/신림동_고시촌_005_52a2f3af.jpg",
]

def main():
    if not SENDER or not APP_PASSWORD or not RECIPIENT:
        raise EnvironmentError(
            "SMTP_SENDER, SMTP_APP_PASSWORD, SMTP_RECIPIENT 환경변수가 필요합니다."
        )

    msg = EmailMessage()
    msg["Subject"] = "3개 이미지 전송"
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg.set_content("요청하신 이미지 3개 첨부합니다.")

    for file_path in FILES:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {file_path}")

        ctype, _ = mimetypes.guess_type(str(p))
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(p, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=p.name,
            )

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)

    print("메일 전송 완료")

if __name__ == "__main__":
    main()