#!/bin/sh

bash db/download_from_gdrive.sh

uvicorn main:app --reload --port 8000