from fastapi import FastAPI
from classifier.routers import pneumonia_router

app = FastAPI()
app.include_router(pneumonia_router.router, prefix='/pneumonia')  # noqa


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Good to go'
