from fastapi import APIRouter, UploadFile, File, HTTPException
from service.recommendation_service import recommend_products
from PIL import Image
from io import BytesIO

router = APIRouter()

@router.post("/recommend_product")
async def recommend_product(file: UploadFile= File(...)):
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        result = recommend_products(image=image)

        return { "result" : result}
    
    except ValueError as e:
        raise(HTTPException(status_code=400, detail=str(e)))
