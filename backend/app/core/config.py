import os


class Settings:
    APP_NAME: str = "VITALIS Near Real-Time rPPG API"
    APP_VERSION: str = "2.1.0"
    UPLOAD_DIR: str = "temp_uploads"
    DEFAULT_MODEL: str = "FacePhys.rlap"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def CORS_ALLOWED_ORIGINS(self) -> list[str]:
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
        if env_origins:
            origins = [origin.strip() for origin in env_origins.split(",")]
            return [origin for origin in origins if origin]

        return [
            "http://localhost:5173",
        ]

    @property
    def upload_dir(self) -> str:
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        return self.UPLOAD_DIR


settings = Settings()
